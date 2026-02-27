"""Сборщик графа агента из декларативных описаний."""
import json
import re
from typing import TypedDict, Annotated, Any
from langchain_core.messages import SystemMessage, AIMessage, HumanMessage
from langchain_core.runnables import RunnableConfig
from langchain_core.tools import tool as langchain_tool
from langgraph.graph import StateGraph, END
from langgraph.graph.message import add_messages
from langgraph.prebuilt import create_react_agent
from langgraph.types import Command
from langgraph.managed import RemainingSteps
from langgraph.errors import GraphRecursionError
from .state import State, MemoryInjection
from .logging_utils import (
    log_state_transition,
    log_memory_snapshot,
    log_warning,
    log_reentry,
)


class AgentState(TypedDict):
    messages: Annotated[list, add_messages]
    memory: dict
    summary: str
    remaining_steps: RemainingSteps


_TRANSITION_KEY = "__transition__"


class AgentGraphBuilder:
    """Строит граф агента из декларативных описаний состояний.

    Переходы между состояниями реализованы через transition tool:
    LLM вызывает transition(next_state=..., summary=...) для перехода.
    """

    def __init__(self, llm, all_tools: dict[str, Any]):
        self.llm = llm
        self.all_tools = all_tools
        self.states: list[State] = []
        self.entry_point: str | None = None
        self._last_state_name: str | None = None
        self._cycle_count: int = 0

    def add_state(self, state: State) -> "AgentGraphBuilder":
        self.states.append(state)
        return self

    def add_states(self, states: list[State]) -> "AgentGraphBuilder":
        self.states.extend(states)
        return self

    def set_entry(self, state_name: str) -> "AgentGraphBuilder":
        self.entry_point = state_name
        return self

    def build(self):
        if not self.entry_point:
            raise ValueError("Точка входа не установлена. Используйте set_entry()")
        if not self.states:
            raise ValueError("Нет состояний. Добавьте хотя бы одно через add_state()")

        workflow = StateGraph(AgentState)

        for state in self.states:
            node_function = self._create_node_function(state)
            workflow.add_node(state.name, node_function)

        workflow.set_entry_point(self.entry_point)

        return workflow.compile()

    def _make_transition_tool(self, state: State):
        allowed = ["stay"] + state.transitions
        allowed_str = ", ".join(f'"{t}"' for t in allowed)
        state_name = state.name

        @langchain_tool
        def transition(reasoning: str, summary: str, next_state: str) -> str:
            """Переход в другое состояние агента. Вызови когда текущая задача завершена.

            Args:
                reasoning: Почему принято решение о переходе.
                summary: Краткое резюме проделанной работы и напутствие следующему состоянию. Обязательно укажи какие ключи сохранены в memory.
                next_state: Куда перейти. "stay" — остаться в текущем состоянии.
            """
            from src.tools.tools import _memory_store

            if next_state not in allowed:
                memory_keys = sorted(list(_memory_store.keys()))
                return (
                    f"Ошибка: переход в '{next_state}' недоступен из состояния '{state_name}'. "
                    f"Доступные переходы: {allowed_str}. "
                    f"Текущие ключи в памяти: {memory_keys}"
                )

            _memory_store[_TRANSITION_KEY] = {
                "reasoning": reasoning,
                "summary": summary,
                "next_state": next_state,
            }

            return (
                f"OK: переход в '{next_state}' подтверждён. "
                f"Не вызывай больше инструментов — напиши финальный ответ."
            )

        return transition

    def _build_transition_prompt_section(self, state: State) -> str:
        allowed = state.transitions
        if not allowed:
            return ""

        transitions_list = "\n".join(f'  - "{t}"' for t in allowed)
        return f"""

## Переход в другое состояние
Когда задача в текущем состоянии выполнена, вызови инструмент transition:
- reasoning: почему переходишь именно туда.
- summary: что было сделано и что нужно сделать в следующем состоянии. Укажи какие ключи сохранены в memory.
- next_state: одно из доступных значений:
  - "stay" — остаться в текущем состоянии (нужно ещё поработать)
{transitions_list}

ВАЖНО: после вызова transition не вызывай другие инструменты — напиши финальный ответ."""

    def _normalize_memory_injection(self, raw: Any, state_name: str) -> dict[str, str | None] | None:
        """Приводит декларацию memory injection к единому формату.

        Поддерживаются:
        - "key"
        - ("key", "if_exists")
        - ("key", "if_exists", "if_missing")
        - {"key": "...", "if_exists": "...", "if_missing": "..."}
        - MemoryInjection(...)
        """
        if isinstance(raw, str):
            key = raw.strip()
            return {"key": key, "if_exists": None, "if_missing": None} if key else None

        if isinstance(raw, MemoryInjection):
            key = (raw.key or "").strip()
            if not key:
                return None
            return {"key": key, "if_exists": raw.if_exists, "if_missing": raw.if_missing}

        if isinstance(raw, (tuple, list)):
            if not raw:
                return None
            key = str(raw[0]).strip()
            if not key:
                return None
            if_exists = str(raw[1]) if len(raw) > 1 and raw[1] is not None else None
            if_missing = str(raw[2]) if len(raw) > 2 and raw[2] is not None else None
            return {"key": key, "if_exists": if_exists, "if_missing": if_missing}

        if isinstance(raw, dict):
            key = str(raw.get("key", "")).strip()
            if not key:
                return None
            if_exists = raw.get("if_exists")
            if_missing = raw.get("if_missing")
            return {
                "key": key,
                "if_exists": str(if_exists) if if_exists is not None else None,
                "if_missing": str(if_missing) if if_missing is not None else None,
            }

        log_warning(
            f"Некорректный memory_injection в '{state_name}': {type(raw).__name__}. "
            "Ожидается str/tuple/list/dict/MemoryInjection."
        )
        return None

    def _build_memory_injection_messages(self, state: State) -> list[HumanMessage]:
        """Формирует служебные сообщения по ключам memory_injections для текущего state."""
        from src.tools.tools import _memory_store

        result: list[HumanMessage] = []
        raw_rules = state.memory_injections or []
        for raw in raw_rules:
            rule = self._normalize_memory_injection(raw, state.name)
            if not rule:
                continue

            key = rule["key"]
            if key in _memory_store:
                value = _memory_store.get(key)
                if_exists = rule["if_exists"] or f"{key}: "
                text = f"{if_exists}{value}"
                result.append(HumanMessage(content=f"Контекст из памяти:\n{text}"))
                continue

            if rule["if_missing"]:
                result.append(HumanMessage(content=f"Контекст из памяти:\n{rule['if_missing']}"))

        return result

    def _summarize_for_reentry(self, messages: list, state_name: str) -> str:
        from src.tools.tools import _memory_store

        memory_keys = sorted(
            k for k in _memory_store.keys() if k != _TRANSITION_KEY
        )

        last_content = ""
        for msg in reversed(messages):
            if isinstance(msg, AIMessage) and msg.content and msg.content.strip():
                last_content = msg.content[:500]
                break

        summary = (
            f"Продолжение работы в состоянии '{state_name}'. "
            f"Последний ответ агента: {last_content}. "
            f"Ключи в памяти: {memory_keys}. "
            f"Используй memory(action='get', key='...') для получения нужных данных."
        )

        return summary

    def _create_node_function(self, state: State):
        state_tools = [self.all_tools[name] for name in state.tools
                       if name in self.all_tools]

        if not state_tools:
            print(f"Предупреждение: состояние '{state.name}' не имеет инструментов")

        transition_tool = self._make_transition_tool(state)
        all_node_tools = state_tools + [transition_tool]

        agent = create_react_agent(self.llm, all_node_tools)

        full_prompt = state.prompt + self._build_transition_prompt_section(state)

        def node_function(state_data: AgentState, config: RunnableConfig | None = None) -> Command:
            self._cycle_count += 1
            prev_state = self._last_state_name or "START"

            if self._last_state_name != state.name:
                log_state_transition(prev_state, state.name)
                self._last_state_name = state.name

            if state.on_enter:
                state_data = state.on_enter(state_data)

            summary = state_data.get("summary", "")
            messages = [SystemMessage(content=full_prompt)]

            existing = state_data.get("messages", [])
            original_query = None
            for msg in existing:
                if isinstance(msg, HumanMessage):
                    original_query = msg
                    break

            if summary:
                if original_query:
                    messages.append(original_query)
                messages.append(HumanMessage(content=f"Контекст из предыдущего состояния:\n{summary}"))
            else:
                for msg in existing:
                    if isinstance(msg, SystemMessage):
                        continue
                    messages.append(msg)

            messages.extend(self._build_memory_injection_messages(state))

            from src.tools.tools import _memory_store
            _memory_store.pop(_TRANSITION_KEY, None)

            inner_config = {"recursion_limit": 25}
            if config:
                inner_config = {**config, "recursion_limit": 25}

            try:
                result = agent.invoke(
                    {"messages": messages},
                    config=inner_config,
                )
            except GraphRecursionError:
                auto_summary = self._summarize_for_reentry(messages, state.name)
                log_reentry(state.name)

                new_state = {
                    "summary": auto_summary,
                    "memory": dict(_memory_store),
                }

                if state.on_exit:
                    new_state = state.on_exit(new_state)

                log_memory_snapshot(state.name, _memory_store, when="exit(reentry)")
                return Command(goto=state.name, update=new_state)

            transition_decision = _memory_store.pop(_TRANSITION_KEY, None)

            new_state: dict[str, Any] = {
                "memory": dict(_memory_store),
            }

            if state.on_exit:
                new_state = state.on_exit(new_state)

            if transition_decision:
                next_target = transition_decision["next_state"]
                transition_summary = transition_decision["summary"]

                memory_keys = sorted(
                    k for k in _memory_store.keys() if k != _TRANSITION_KEY
                )
                context = (
                    f"{transition_summary}\n"
                    f"Ключи в памяти: {memory_keys}. "
                    f"Используй memory(action='get', key='...') для получения данных."
                )

                log_memory_snapshot(state.name, _memory_store, when="exit")

                if next_target == "stay":
                    new_state["summary"] = context
                    return Command(goto=state.name, update=new_state)

                if next_target == "END":
                    last_msg = self._get_last_ai_message(result["messages"])
                    new_state["messages"] = [last_msg] if last_msg else result["messages"]
                    new_state["summary"] = ""
                    return Command(goto=END, update=new_state)

                new_state["summary"] = context
                return Command(goto=next_target, update=new_state)

            fallback = self._parse_transition_from_text(
                result["messages"], state.transitions
            )
            if fallback:
                next_target = fallback["next_state"]
                transition_summary = fallback["summary"]

                memory_keys = sorted(
                    k for k in _memory_store.keys() if k != _TRANSITION_KEY
                )
                context = (
                    f"{transition_summary}\n"
                    f"Ключи в памяти: {memory_keys}. "
                    f"Используй memory(action='get', key='...') для получения данных."
                )

                log_memory_snapshot(state.name, _memory_store, when="exit(fallback)")

                if next_target == "stay":
                    new_state["summary"] = context
                    return Command(goto=state.name, update=new_state)

                if next_target == "END":
                    last_msg = self._get_last_ai_message(result["messages"])
                    new_state["messages"] = [last_msg] if last_msg else result["messages"]
                    new_state["summary"] = ""
                    return Command(goto=END, update=new_state)

                new_state["summary"] = context
                return Command(goto=next_target, update=new_state)

            log_warning(f"Transition не вызван в '{state.name}', остаёмся")

            new_state["summary"] = self._summarize_for_reentry(
                result["messages"], state.name
            )
            log_memory_snapshot(state.name, _memory_store, when="exit(no_transition)")
            return Command(goto=state.name, update=new_state)

        return node_function

    def _parse_transition_from_text(self, messages: list, allowed: list[str]) -> dict | None:
        """Fallback: ищет transition-решение в тексте последнего AIMessage."""
        all_allowed = allowed + ["stay"]

        for msg in reversed(messages):
            if not isinstance(msg, AIMessage) or not msg.content:
                continue
            content = msg.content

            for match in re.finditer(r'\{[^{}]*"next_state"[^{}]*\}', content):
                try:
                    data = json.loads(match.group())
                except json.JSONDecodeError:
                    cleaned = match.group()
                    try:
                        data = json.loads(cleaned.replace('\\"', '"'))
                    except (json.JSONDecodeError, KeyError):
                        continue

                next_state = data.get("next_state", "")
                if next_state in all_allowed:
                    return {
                        "next_state": next_state,
                        "summary": data.get("summary", ""),
                        "reasoning": data.get("reasoning", ""),
                    }

            for target in allowed:
                pattern = rf'"next_state"\s*:\s*"{re.escape(target)}"'
                if re.search(pattern, content):
                    summary_match = re.search(r'"summary"\s*:\s*"([^"]*(?:\\"[^"]*)*)"', content)
                    return {
                        "next_state": target,
                        "summary": summary_match.group(1).replace('\\"', '"') if summary_match else "",
                        "reasoning": "",
                    }
        return None

    def _get_last_ai_message(self, messages: list) -> AIMessage | None:
        for msg in reversed(messages):
            if isinstance(msg, AIMessage) and msg.content and msg.content.strip():
                if not msg.tool_calls:
                    return AIMessage(content=msg.content)
        return None

    def visualize(self) -> str:
        lines = ["Граф агента:", ""]

        lines.append("Состояния:")
        for s in self.states:
            entry_marker = " (entry)" if s.name == self.entry_point else ""
            lines.append(f"  - {s.name}{entry_marker}")
            lines.append(f"    Инструменты: {', '.join(s.tools)}")
            if s.transitions:
                lines.append(f"    Переходы: {', '.join(s.transitions)}")
            if s.description:
                lines.append(f"    Описание: {s.description}")

        return "\n".join(lines)
