"""Сборщик графа агента из декларативных описаний."""
import json
import re
from typing import TypedDict, Annotated, Any
from langchain_core.messages import SystemMessage, AIMessage, HumanMessage, ToolMessage
from langchain_core.tools import tool as langchain_tool
from langgraph.graph import StateGraph, END
from langgraph.graph.message import add_messages
from langgraph.prebuilt import create_react_agent
from langgraph.types import Command
from langgraph.managed import RemainingSteps
from langgraph.errors import GraphRecursionError
from .state import State
from .debug import log_prompts_enabled, log_event, serialize_messages


class AgentState(TypedDict):
    """Состояние агента в графе."""
    messages: Annotated[list, add_messages]
    memory: dict
    summary: str
    remaining_steps: RemainingSteps


# Ключ в памяти, куда transition tool записывает решение
_TRANSITION_KEY = "__transition__"


class AgentGraphBuilder:
    """Строит граф агента из декларативных описаний состояний.
    
    Переходы между состояниями реализованы через transition tool:
    LLM вызывает transition(next_state=..., summary=...) для перехода.
    Допустимые переходы определяются в State.transitions.
    
    Примеры:
        builder = AgentGraphBuilder(llm, tools_dict)
        graph = (builder
            .add_state(State(name="work", tools=["calc"], prompt="...", transitions=["summarize"]))
            .add_state(State(name="summarize", tools=["memory"], prompt="...", transitions=["END"]))
            .set_entry("work")
            .build())
    """
    
    def __init__(self, llm, all_tools: dict[str, Any]):
        self.llm = llm
        self.all_tools = all_tools
        self.states: list[State] = []
        self.entry_point: str | None = None
        self._last_state_name: str | None = None
        self._cycle_count: int = 0
    
    def add_state(self, state: State) -> 'AgentGraphBuilder':
        """Добавляет состояние в граф."""
        self.states.append(state)
        return self
    
    def add_states(self, states: list[State]) -> 'AgentGraphBuilder':
        """Добавляет несколько состояний в граф."""
        self.states.extend(states)
        return self
    
    def set_entry(self, state_name: str) -> 'AgentGraphBuilder':
        """Устанавливает точку входа в граф."""
        self.entry_point = state_name
        return self
    
    def build(self):
        """Собирает и компилирует граф."""
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
        """Создаёт transition tool для конкретного состояния.
        
        Tool знает допустимые переходы и возвращает ошибку с подсказкой,
        если LLM указала невалидное состояние.
        """
        allowed = ["stay"] + state.transitions
        allowed_str = ", ".join(f'"{t}"' for t in allowed)
        state_name = state.name

        @langchain_tool
        def transition(next_state: str, summary: str, reasoning: str) -> str:
            """Переход в другое состояние агента. Вызови когда текущая задача завершена.
            
            Args:
                next_state: Куда перейти. "stay" — остаться в текущем состоянии.
                summary: Краткое резюме проделанной работы и напутствие следующему состоянию. Обязательно укажи какие ключи сохранены в memory.
                reasoning: Почему принято решение о переходе.
            """
            from tools.tools import _memory_store
            
            if next_state not in allowed:
                memory_keys = sorted(list(_memory_store.keys()))
                return (
                    f"Ошибка: переход в '{next_state}' недоступен из состояния '{state_name}'. "
                    f"Доступные переходы: {allowed_str}. "
                    f"Текущие ключи в памяти: {memory_keys}"
                )
            
            _memory_store[_TRANSITION_KEY] = {
                "next_state": next_state,
                "summary": summary,
                "reasoning": reasoning,
            }
            
            memory_keys = sorted(
                k for k in _memory_store.keys() if k != _TRANSITION_KEY
            )
            
            log_event(
                "transition_tool",
                {
                    "from_state": state_name,
                    "next_state": next_state,
                    "summary": summary,
                    "reasoning": reasoning,
                    "memory_keys": memory_keys,
                },
            )
            
            return (
                f"OK: переход в '{next_state}' подтверждён. "
                f"Не вызывай больше инструментов — напиши финальный ответ."
            )
        
        return transition
    
    def _build_transition_prompt_section(self, state: State) -> str:
        """Генерирует секцию промпта про переходы для состояния."""
        allowed = state.transitions
        if not allowed:
            return ""
        
        transitions_list = "\n".join(f'  - "{t}"' for t in allowed)
        return f"""

## Переход в другое состояние
Когда задача в текущем состоянии выполнена, вызови инструмент transition:
- next_state: одно из доступных значений:
  - "stay" — остаться в текущем состоянии (нужно ещё поработать)
{transitions_list}
- summary: что было сделано и что нужно сделать в следующем состоянии. Укажи какие ключи сохранены в memory.
- reasoning: почему переходишь именно туда.

ВАЖНО: после вызова transition не вызывай другие инструменты — напиши финальный ответ."""

    def _summarize_for_reentry(self, messages: list, state_name: str) -> str:
        """Суммаризация при повторном входе в состояние (лимит шагов)."""
        from tools.tools import _memory_store
        
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
        
        log_event(
            "auto_summarization",
            {
                "state": state_name,
                "memory_keys": memory_keys,
                "summary_length": len(summary),
            },
        )
        
        return summary
    
    def _create_node_function(self, state: State):
        """Создает функцию узла для состояния."""
        state_tools = [self.all_tools[name] for name in state.tools 
                      if name in self.all_tools]
        
        if not state_tools:
            print(f"Предупреждение: состояние '{state.name}' не имеет инструментов")
        
        transition_tool = self._make_transition_tool(state)
        all_node_tools = state_tools + [transition_tool]
        
        agent = create_react_agent(self.llm, all_node_tools)
        
        full_prompt = state.prompt + self._build_transition_prompt_section(state)

        def node_function(state_data: AgentState) -> Command:
            """Функция узла состояния. Возвращает Command для маршрутизации."""
            self._cycle_count += 1
            step = self._cycle_count
            prev_state = self._last_state_name or "START"
            
            if self._last_state_name != state.name:
                if log_prompts_enabled():
                    print(f"[STATE] {prev_state} -> {state.name}")
                self._last_state_name = state.name
            
            remaining = state_data.get("remaining_steps", 50)
            
            log_event(
                "state_transition",
                {
                    "from_state": prev_state,
                    "to_state": state.name,
                    "message_count": len(state_data.get("messages", [])),
                    "memory_keys": sorted(list((state_data.get("memory") or {}).keys())),
                    "remaining_steps": remaining,
                },
                step=step,
                state=state.name,
            )
            
            if state.on_enter:
                state_data = state.on_enter(state_data)
            
            # Собираем сообщения: system prompt + original query + context
            summary = state_data.get("summary", "")
            messages = [SystemMessage(content=full_prompt)]
            
            # Всегда ищем оригинальный запрос пользователя
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
            
            log_event(
                "llm_request",
                {
                    "messages": serialize_messages(messages),
                    "has_summary": bool(summary),
                },
                step=step,
                state=state.name,
            )
            
            # Очищаем предыдущий transition из памяти
            from tools.tools import _memory_store
            _memory_store.pop(_TRANSITION_KEY, None)
            
            # Запускаем ReAct-агент
            try:
                result = agent.invoke(
                    {"messages": messages},
                    config={"recursion_limit": 25},
                )
            except GraphRecursionError:
                auto_summary = self._summarize_for_reentry(messages, state.name)
                
                if log_prompts_enabled():
                    print(f"[REENTRY] Лимит шагов, суммаризация -> {state.name}")
                
                new_state = {
                    "summary": auto_summary,
                    "memory": dict(_memory_store),
                }
                
                if state.on_exit:
                    new_state = state.on_exit(new_state)
                
                return Command(goto=state.name, update=new_state)
            
            log_event(
                "llm_response",
                {
                    "messages": serialize_messages(result.get("messages", [])),
                },
                step=step,
                state=state.name,
            )
            
            # Проверяем, был ли вызван transition tool
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
            
            # Fallback: модель могла написать transition как текст
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
            
            # Transition не вызван — остаёмся в текущем состоянии
            if log_prompts_enabled():
                print(f"[WARN] Transition не вызван в '{state.name}', остаёмся")
            
            new_state["summary"] = self._summarize_for_reentry(
                result["messages"], state.name
            )
            return Command(goto=state.name, update=new_state)
        
        return node_function
    
    def _parse_transition_from_text(self, messages: list, allowed: list[str]) -> dict | None:
        """Fallback: ищет transition-решение в тексте последнего AIMessage.
        
        Некоторые модели (особенно локальные) пишут вызов transition как текст
        вместо function call. Пытаемся извлечь JSON с next_state.
        """
        all_allowed = allowed + ["stay"]
        
        for msg in reversed(messages):
            if not isinstance(msg, AIMessage) or not msg.content:
                continue
            content = msg.content
            
            # Ищем все JSON-подобные фрагменты с next_state
            for match in re.finditer(r'\{[^{}]*"next_state"[^{}]*\}', content):
                try:
                    data = json.loads(match.group())
                except json.JSONDecodeError:
                    # JSON с escaped кавычками — чистим и пробуем снова
                    cleaned = match.group()
                    try:
                        data = json.loads(cleaned.replace('\\"', '"'))
                    except (json.JSONDecodeError, KeyError):
                        continue
                
                next_state = data.get("next_state", "")
                if next_state in all_allowed:
                    if log_prompts_enabled():
                        print(f"[FALLBACK] Transition из текста: {next_state}")
                    return {
                        "next_state": next_state,
                        "summary": data.get("summary", ""),
                        "reasoning": data.get("reasoning", ""),
                    }
            
            # Ещё один fallback: ищем "next_state":"END" в произвольном тексте
            for target in allowed:
                pattern = rf'"next_state"\s*:\s*"{re.escape(target)}"'
                if re.search(pattern, content):
                    if log_prompts_enabled():
                        print(f"[FALLBACK] Transition из текста (regex): {target}")
                    summary_match = re.search(r'"summary"\s*:\s*"([^"]*(?:\\"[^"]*)*)"', content)
                    return {
                        "next_state": target,
                        "summary": summary_match.group(1).replace('\\"', '"') if summary_match else "",
                        "reasoning": "",
                    }
        return None
    
    def _get_last_ai_message(self, messages: list) -> AIMessage | None:
        """Находит последнее осмысленное AIMessage."""
        for msg in reversed(messages):
            if isinstance(msg, AIMessage) and msg.content and msg.content.strip():
                if not msg.tool_calls:
                    return AIMessage(content=msg.content)
        return None
    
    def visualize(self) -> str:
        """Возвращает текстовое представление графа."""
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
