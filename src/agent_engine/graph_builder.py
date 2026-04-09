"""Сборщик графа агента из декларативных описаний."""
import json
import re
from pathlib import Path
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
from .state import State, MemoryInjection, AutoTransitionRule
from .logging_utils import (
    log_state_transition,
    log_memory_snapshot,
    log_warning,
    log_reentry,
    log_transition_mode,
)


class AgentState(TypedDict):
    messages: Annotated[list, add_messages]
    memory: dict
    summary: str
    remaining_steps: RemainingSteps


_TRANSITION_KEY = "__transition__"


class EarlyBreakTransition(Exception):
    """Сигнал штатного раннего выхода после успешного transition."""


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

    def register_sub_agents(
        self,
        owner_agent_name: str,
        sub_agents: list[str] | tuple[str, ...] | None,
    ) -> None:
        """Регистрирует под-агентов для мультиагентного сценария.

        Правило декларативное: агент задает `sub_agents = [...]`,
        а движок централизованно валидирует и регистрирует их.
        """
        if sub_agents is None:
            return
        if not isinstance(sub_agents, (list, tuple)):
            raise TypeError("sub_agents должен быть list или tuple")

        from src.agents import build_agent, list_agents
        from src.tools.tools import register_agent

        available_agents = set(list_agents())
        for sub_agent_name in sub_agents:
            if not isinstance(sub_agent_name, str) or not sub_agent_name.strip():
                raise ValueError("Все элементы sub_agents должны быть непустыми строками")
            if sub_agent_name == owner_agent_name:
                raise ValueError(
                    f"Нельзя регистрировать '{owner_agent_name}' как подчиненного самому себе"
                )
            if sub_agent_name not in available_agents:
                available = ", ".join(sorted(available_agents))
                raise ValueError(
                    f"Подчиненный агент '{sub_agent_name}' не найден. Доступные агенты: {available}"
                )
            register_agent(sub_agent_name, build_agent(sub_agent_name, self.llm))

    def build(self, checkpointer=None):
        if not self.entry_point:
            raise ValueError("Точка входа не установлена. Используйте set_entry()")
        if not self.states:
            raise ValueError("Нет состояний. Добавьте хотя бы одно через add_state()")

        workflow = StateGraph(AgentState)

        for state in self.states:
            node_function = self._create_node_function(state)
            workflow.add_node(state.name, node_function)

        workflow.set_entry_point(self.entry_point)

        return workflow.compile(checkpointer=checkpointer)

    def _make_transition_tool(self, state: State):
        allowed = ["stay"] + state.transitions
        allowed_str = ", ".join(f'"{t}"' for t in allowed)
        state_name = state.name

        @langchain_tool
        def transition(reasoning: str = "", summary: str = "", next_state: str = "") -> str:
            """Переход в другое состояние агента. Вызови когда текущая задача завершена.

            Args:
                reasoning: Почему принято решение о переходе.
                summary: Краткое резюме проделанной работы и напутствие следующему состоянию.
                next_state: Куда перейти. "stay" — остаться в текущем состоянии.
            """
            from src.tools.tools import _memory_store

            if not next_state:
                return (
                    "Ошибка: next_state обязателен. "
                    f"Доступные переходы: {allowed_str}."
                )

            if next_state not in allowed:
                memory_keys = sorted(list(_memory_store.keys()))
                return (
                    f"Ошибка: переход в '{next_state}' недоступен из состояния '{state_name}'. "
                    f"Доступные переходы: {allowed_str}. "
                    f"Текущие ключи в памяти: {memory_keys}"
                )

            if state.require_transition_summary and not summary.strip():
                return (
                    "Ошибка: в этом состоянии summary обязателен. "
                    "Передай краткое резюме и какие ключи сохранены в memory."
                )

            if state.require_transition_reasoning and not reasoning.strip():
                return "Ошибка: в этом состоянии reasoning обязателен."

            _memory_store[_TRANSITION_KEY] = {
                "reasoning": reasoning,
                "summary": summary,
                "next_state": next_state,
            }

            if state.early_break and next_state != "stay":
                raise EarlyBreakTransition(
                    f"EARLY_BREAK transition={state_name}->{next_state}"
                )

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
        mode = "FAST" if state.fast_transition else "FULL"
        field_rules = [
            "- next_state: ОБЯЗАТЕЛЕН, укажи состояние из списка ниже.",
            (
                "- reasoning: ОБЯЗАТЕЛЕН."
                if state.require_transition_reasoning
                else "- reasoning: опционален."
            ),
            (
                "- summary: ОБЯЗАТЕЛЕН, укажи что сделано и какие ключи сохранены в memory."
                if state.require_transition_summary
                else "- summary: опционален."
            ),
        ]
        if state.fast_transition:
            field_rules.append(
                "- Быстрый режим: старайся вызывать transition сразу после достижения результата."
            )
        return f"""

## Переход в другое состояние
Когда задача в текущем состоянии выполнена, вызови инструмент transition:
Режим: {mode}
{chr(10).join(field_rules)}
Доступные значения next_state:
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

    def _normalize_auto_transition_rule(self, raw: Any, state_name: str) -> dict[str, Any] | None:
        if isinstance(raw, AutoTransitionRule):
            return {
                "next_state": raw.next_state,
                "summary": raw.summary,
                "memory_has_all": list(raw.memory_has_all or []),
                "memory_equals": dict(raw.memory_equals or {}),
                "memory_regex": dict(raw.memory_regex or {}),
                "file_exists": list(raw.file_exists or []),
            }
        if isinstance(raw, dict):
            next_state = str(raw.get("next_state", "")).strip()
            if not next_state:
                return None
            return {
                "next_state": next_state,
                "summary": str(raw.get("summary", "")),
                "memory_has_all": list(raw.get("memory_has_all", []) or []),
                "memory_equals": dict(raw.get("memory_equals", {}) or {}),
                "memory_regex": dict(raw.get("memory_regex", {}) or {}),
                "file_exists": list(raw.get("file_exists", []) or []),
            }
        log_warning(
            f"Некорректный auto_transition в '{state_name}': {type(raw).__name__}. "
            "Ожидается dict или AutoTransitionRule."
        )
        return None

    def _match_auto_transition(self, state: State, memory_store: dict[str, Any]) -> dict[str, str] | None:
        rules = state.auto_transitions or []
        if not rules:
            return None

        allowed = {"stay"} | set(state.transitions)
        for raw in rules:
            rule = self._normalize_auto_transition_rule(raw, state.name)
            if not rule:
                continue

            next_state = rule["next_state"]
            if next_state not in allowed:
                log_warning(
                    f"auto_transition в '{state.name}' ведёт в '{next_state}', "
                    f"но доступно только: {sorted(list(allowed))}"
                )
                continue

            has_all = True
            for key in rule["memory_has_all"]:
                if key not in memory_store:
                    has_all = False
                    break
            if not has_all:
                continue

            equals_ok = True
            for key, expected in rule["memory_equals"].items():
                if memory_store.get(key) != expected:
                    equals_ok = False
                    break
            if not equals_ok:
                continue

            regex_ok = True
            for key, pattern_value in rule["memory_regex"].items():
                current = str(memory_store.get(key, ""))
                patterns = (
                    pattern_value
                    if isinstance(pattern_value, (list, tuple))
                    else [pattern_value]
                )
                matched_any = False
                for pattern in patterns:
                    try:
                        if re.search(str(pattern), current):
                            matched_any = True
                            break
                    except re.error:
                        log_warning(
                            f"Некорректный regex в auto_transition '{state.name}': "
                            f"key='{key}', pattern='{pattern}'"
                        )
                if not matched_any:
                    regex_ok = False
                    break
            if not regex_ok:
                continue

            files_ok = True
            for p in rule["file_exists"]:
                if not Path(str(p)).exists():
                    files_ok = False
                    break
            if not files_ok:
                continue

            return {
                "next_state": next_state,
                "summary": rule["summary"],
            }
        return None

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

            # Синхронизируем _memory_store из стейта.
            # Критично при resume/fork: checkpoint содержит актуальную память,
            # а глобальный _memory_store может быть устаревшим или пустым.
            from src.tools.tools import _memory_store
            state_memory = state_data.get("memory") or {}
            _memory_store.clear()
            _memory_store.update(state_memory)
            _memory_store.pop(_TRANSITION_KEY, None)

            auto_transition = self._match_auto_transition(state, _memory_store)
            if auto_transition:
                next_target = auto_transition["next_state"]
                transition_summary = auto_transition.get("summary", "")
                memory_keys = sorted(
                    k for k in _memory_store.keys() if k != _TRANSITION_KEY
                )
                context = (
                    f"{transition_summary}\n"
                    f"Ключи в памяти: {memory_keys}. "
                    f"Используй memory(action='get', key='...') для получения данных."
                ).strip()

                new_state: dict[str, Any] = {"memory": dict(_memory_store)}
                if state.on_exit:
                    new_state = state.on_exit(new_state)
                log_transition_mode("auto_transition", state.name, next_target)
                log_memory_snapshot(state.name, _memory_store, when="exit(auto)")

                if next_target == "stay":
                    new_state["summary"] = context
                    return Command(goto=state.name, update=new_state)
                if next_target == "END":
                    new_state["summary"] = ""
                    return Command(goto=END, update=new_state)

                new_state["summary"] = context
                return Command(goto=next_target, update=new_state)

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

            inner_config = {"recursion_limit": 25}
            if config:
                inner_config = {**config, "recursion_limit": 25}

            try:
                result = agent.invoke(
                    {"messages": messages},
                    config=inner_config,
                )
            except EarlyBreakTransition:
                transition_decision = _memory_store.pop(_TRANSITION_KEY, None)
                new_state: dict[str, Any] = {"memory": dict(_memory_store)}
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
                    log_transition_mode("early_break", state.name, next_target)
                    log_memory_snapshot(state.name, _memory_store, when="exit(early_break)")

                    if next_target == "stay":
                        new_state["summary"] = context
                        return Command(goto=state.name, update=new_state)
                    if next_target == "END":
                        new_state["summary"] = ""
                        return Command(goto=END, update=new_state)

                    new_state["summary"] = context
                    return Command(goto=next_target, update=new_state)

                log_warning(f"Early break без transition в '{state.name}', остаёмся")
                new_state["summary"] = self._summarize_for_reentry(messages, state.name)
                log_memory_snapshot(state.name, _memory_store, when="exit(early_break_no_transition)")
                return Command(goto=state.name, update=new_state)
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

                log_transition_mode("transition_tool", state.name, next_target)
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

                log_transition_mode("transition_tool", state.name, next_target)
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

    def _classify_tool_origin(self, tool_name: str) -> str:
        """Определяет происхождение инструмента для preflight-визуализации."""
        tool_obj = self.all_tools.get(tool_name)
        if tool_obj is None:
            return "missing"
        if tool_name == "transition":
            return "system"

        module_name = getattr(tool_obj, "__module__", "") or ""
        raw_func = getattr(tool_obj, "func", None)
        if raw_func is not None:
            module_name = getattr(raw_func, "__module__", "") or module_name

        if module_name.startswith("src.agents."):
            return "local"
        return "shared"

    def _collect_preflight_issues(self) -> dict[str, list[str]]:
        states = self.states
        state_names = [s.name for s in states]
        unique_names = set(state_names)

        duplicate_state_names = sorted({name for name in state_names if state_names.count(name) > 1})

        invalid_transitions: list[str] = []
        adjacency: dict[str, list[str]] = {s.name: [] for s in states}
        for s in states:
            for target in s.transitions:
                if target == "END":
                    continue
                if target not in unique_names:
                    invalid_transitions.append(f"{s.name} -> {target}")
                    continue
                adjacency[s.name].append(target)

        missing_tools: list[str] = []
        for s in states:
            missing = [tool_name for tool_name in s.tools if tool_name not in self.all_tools]
            if missing:
                missing_tools.append(f"{s.name}: {', '.join(missing)}")

        entry_point_issues: list[str] = []
        if not self.entry_point:
            entry_point_issues.append("entry_point не задан")
        elif self.entry_point not in unique_names:
            entry_point_issues.append(f"entry_point '{self.entry_point}' отсутствует в states")

        unreachable_states: list[str] = []
        if self.entry_point and self.entry_point in unique_names:
            visited: set[str] = set()
            stack = [self.entry_point]
            while stack:
                current = stack.pop()
                if current in visited:
                    continue
                visited.add(current)
                for nxt in adjacency.get(current, []):
                    if nxt not in visited:
                        stack.append(nxt)
            unreachable_states = sorted(unique_names - visited)

        return {
            "entry_point_issues": entry_point_issues,
            "duplicate_state_names": duplicate_state_names,
            "invalid_transitions": sorted(invalid_transitions),
            "missing_tools": missing_tools,
            "unreachable_states": unreachable_states,
        }

    def visualize(self) -> str:
        from src.tools.tools import _memory_store

        lines = ["Граф агента (preflight):", ""]

        unique_tools = sorted({tool_name for s in self.states for tool_name in s.tools})
        memory_keys = sorted(_memory_store.keys())
        issues = self._collect_preflight_issues()

        lines.append("Сводка:")
        lines.append(f"  - Entry point: {self.entry_point or 'не задан'}")
        lines.append(f"  - Состояний: {len(self.states)}")
        lines.append(f"  - Уникальных тулов: {len(unique_tools)}")
        lines.append(f"  - Ключей в памяти сейчас: {len(memory_keys)}")
        if memory_keys:
            lines.append(f"  - Память: {', '.join(memory_keys)}")

        lines.append("")
        lines.append("Состояния:")
        for s in self.states:
            entry_marker = " (entry)" if s.name == self.entry_point else ""
            lines.append(f"  - {s.name}{entry_marker}")
            if s.description:
                lines.append(f"    Описание: {s.description}")

            transitions = ", ".join(s.transitions) if s.transitions else "нет"
            lines.append(f"    Переходы: {transitions}")
            lines.append(
                "    Режим перехода: "
                f"fast_transition={'yes' if s.fast_transition else 'no'}, "
                f"early_break={'yes' if s.early_break else 'no'}"
            )
            if s.auto_transitions:
                rendered_rules = []
                for raw in s.auto_transitions:
                    rule = self._normalize_auto_transition_rule(raw, s.name)
                    if not rule:
                        continue
                    rendered_rules.append(
                        f"{rule['next_state']} "
                        f"(has={rule['memory_has_all']}, eq={rule['memory_equals']}, "
                        f"regex={rule['memory_regex']}, files={rule['file_exists']})"
                    )
                lines.append(
                    "    Auto transitions: "
                    + (", ".join(rendered_rules) if rendered_rules else "нет")
                )
            else:
                lines.append("    Auto transitions: нет")

            if s.tools:
                grouped_tools: dict[str, list[str]] = {
                    "shared": [],
                    "local": [],
                    "missing": [],
                    "other": [],
                }
                for tool_name in s.tools:
                    origin = self._classify_tool_origin(tool_name)
                    if origin in grouped_tools:
                        grouped_tools[origin].append(tool_name)
                    else:
                        grouped_tools["other"].append(f"{tool_name} [{origin}]")

                if grouped_tools["shared"]:
                    lines.append(f"    Инструменты (shared): {', '.join(grouped_tools['shared'])}")
                if grouped_tools["local"]:
                    lines.append(f"    Инструменты (local): {', '.join(grouped_tools['local'])}")
                if grouped_tools["missing"]:
                    lines.append(f"    Инструменты (missing): {', '.join(grouped_tools['missing'])}")
                if grouped_tools["other"]:
                    lines.append(f"    Инструменты (other): {', '.join(grouped_tools['other'])}")
            else:
                lines.append("    Инструменты: нет")

            if s.memory_injections:
                keys = []
                for raw in s.memory_injections:
                    normalized = self._normalize_memory_injection(raw, s.name)
                    if normalized:
                        keys.append(normalized["key"])
                lines.append(f"    Memory injections: {', '.join(keys) if keys else 'нет'}")
            else:
                lines.append("    Memory injections: нет")

        lines.append("")
        lines.append("Проверки конфигурации:")
        if not any(issues.values()):
            lines.append("  ✓ Ошибок конфигурации не найдено")
        else:
            if issues["entry_point_issues"]:
                lines.append(f"  ! Entry point: {'; '.join(issues['entry_point_issues'])}")
            if issues["duplicate_state_names"]:
                lines.append(f"  ! Дубли состояний: {', '.join(issues['duplicate_state_names'])}")
            if issues["invalid_transitions"]:
                lines.append(f"  ! Невалидные переходы: {', '.join(issues['invalid_transitions'])}")
            if issues["missing_tools"]:
                lines.append(f"  ! Отсутствующие тулы: {'; '.join(issues['missing_tools'])}")
            if issues["unreachable_states"]:
                lines.append(
                    f"  ! Недостижимые состояния (в них агент никогда не войдет): "
                    f"{', '.join(issues['unreachable_states'])}"
                )

        return "\n".join(lines)
