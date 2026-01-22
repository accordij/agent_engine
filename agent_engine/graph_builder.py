"""Сборщик графа агента из декларативных описаний."""
from typing import TypedDict, Annotated
from langchain_core.messages import SystemMessage
from langgraph.graph import StateGraph, END
from langgraph.graph.message import add_messages
from langgraph.prebuilt import create_react_agent
from .state import State, Transition


class AgentState(TypedDict):
    """Состояние агента в графе."""
    messages: Annotated[list, add_messages]
    memory: dict


class AgentGraphBuilder:
    """Строит граф агента из декларативных описаний состояний и переходов.
    
    Примеры:
        # Простой случай
        builder = AgentGraphBuilder(llm, tools_dict)
        graph = (builder
            .add_state(work_state)
            .add_state(summarize_state)
            .add_transition(work_to_summarize)
            .add_transition(summarize_to_end)
            .set_entry("work")
            .build())
        
        # Или с списками
        builder = AgentGraphBuilder(llm, tools_dict)
        builder.add_states([work_state, summarize_state])
        builder.add_transitions(all_transitions)
        builder.set_entry("work")
        graph = builder.build()
    
    Методы можно вызывать цепочкой (fluent interface).
    """
    
    def __init__(self, llm, all_tools: dict[str, any]):
        """Инициализирует сборщик графа.
        
        Args:
            llm: Языковая модель
            all_tools: Словарь {имя_инструмента: объект_инструмента}
        """
        self.llm = llm
        self.all_tools = all_tools
        self.states: list[State] = []
        self.transitions: list[Transition] = []
        self.entry_point: str | None = None
    
    def add_state(self, state: State) -> 'AgentGraphBuilder':
        """Добавляет состояние в граф.
        
        Args:
            state: Описание состояния
            
        Returns:
            self (для цепочки вызовов)
        """
        self.states.append(state)
        return self
    
    def add_states(self, states: list[State]) -> 'AgentGraphBuilder':
        """Добавляет несколько состояний в граф.
        
        Args:
            states: Список описаний состояний
            
        Returns:
            self (для цепочки вызовов)
        """
        self.states.extend(states)
        return self
    
    def add_transition(self, transition: Transition) -> 'AgentGraphBuilder':
        """Добавляет переход между состояниями.
        
        Args:
            transition: Описание перехода
            
        Returns:
            self (для цепочки вызовов)
        """
        self.transitions.append(transition)
        return self
    
    def add_transitions(self, transitions: list[Transition]) -> 'AgentGraphBuilder':
        """Добавляет несколько переходов.
        
        Args:
            transitions: Список описаний переходов
            
        Returns:
            self (для цепочки вызовов)
        """
        self.transitions.extend(transitions)
        return self
    
    def set_entry(self, state_name: str) -> 'AgentGraphBuilder':
        """Устанавливает точку входа в граф.
        
        Args:
            state_name: Имя состояния, с которого начинается граф
            
        Returns:
            self (для цепочки вызовов)
        """
        self.entry_point = state_name
        return self
    
    def build(self):
        """Собирает и компилирует граф из описаний.
        
        Returns:
            Скомпилированный граф LangGraph
            
        Raises:
            ValueError: Если не указана точка входа или есть ошибки в конфигурации
        """
        if not self.entry_point:
            raise ValueError("Точка входа не установлена. Используйте set_entry()")
        
        if not self.states:
            raise ValueError("Нет состояний. Добавьте хотя бы одно состояние через add_state()")
        
        # Создаем граф
        workflow = StateGraph(AgentState)
        
        # Создаем узлы для каждого состояния
        for state in self.states:
            node_function = self._create_node_function(state)
            workflow.add_node(state.name, node_function)
        
        # Устанавливаем точку входа
        workflow.set_entry_point(self.entry_point)
        
        # Добавляем переходы
        for transition in self.transitions:
            self._add_transition_to_workflow(workflow, transition)
        
        # Компилируем и возвращаем
        return workflow.compile()
    
    def _create_node_function(self, state: State):
        """Создает функцию узла для состояния.
        
        Args:
            state: Описание состояния
            
        Returns:
            Функция узла для LangGraph
        """
        # Получаем инструменты для этого состояния
        state_tools = [self.all_tools[name] for name in state.tools 
                      if name in self.all_tools]
        
        if not state_tools:
            print(f"⚠️ Предупреждение: состояние '{state.name}' не имеет инструментов")
        
        # Создаем ReAct агента для этого состояния
        agent = create_react_agent(self.llm, state_tools)
        
        def node_function(agent_state: AgentState) -> AgentState:
            """Функция узла состояния."""
            # On enter hook
            if state.on_enter:
                agent_state = state.on_enter(agent_state)
            
            # Добавляем системный промпт, если его еще нет
            messages = agent_state['messages']
            has_system = any(isinstance(msg, SystemMessage) for msg in messages)
            
            if not has_system:
                messages = [SystemMessage(content=state.prompt)] + messages
            
            # Вызываем агента
            result = agent.invoke({'messages': messages})
            
            # Синхронизируем глобальную память инструмента с состоянием
            from tools.tools import _memory_store
            
            # Формируем новое состояние
            new_state = {
                'messages': result['messages'],
                'memory': dict(_memory_store)  # Копируем текущее состояние памяти
            }
            
            # On exit hook
            if state.on_exit:
                new_state = state.on_exit(new_state)
            
            return new_state
        
        return node_function
    
    def _add_transition_to_workflow(self, workflow: StateGraph, transition: Transition):
        """Добавляет переход в workflow.
        
        Args:
            workflow: Граф LangGraph
            transition: Описание перехода
        """
        # Случай 1: Роутер (несколько возможных целевых состояний)
        if transition.routes:
            if not transition.condition:
                raise ValueError(
                    f"Переход-роутер из '{transition.from_state}' требует condition"
                )
            
            workflow.add_conditional_edges(
                transition.from_state,
                transition.condition,
                transition.routes
            )
        
        # Случай 2: Условный переход (возврат в себя или переход дальше)
        elif transition.condition and transition.to_state:
            def make_conditional(trans):
                def condition_wrapper(state: AgentState) -> str:
                    # Если условие True → переходим, иначе остаемся
                    result = trans.condition(state)
                    if result:
                        return trans.to_state
                    else:
                        return trans.from_state
                return condition_wrapper
            
            # Если целевое состояние - END, используем константу END
            target = END if transition.to_state == "END" else transition.to_state
            
            workflow.add_conditional_edges(
                transition.from_state,
                make_conditional(transition),
                {
                    transition.from_state: transition.from_state,
                    transition.to_state: target
                }
            )
        
        # Случай 3: Безусловный переход
        elif transition.to_state:
            if transition.to_state == "END":
                workflow.add_edge(transition.from_state, END)
            else:
                workflow.add_edge(transition.from_state, transition.to_state)
        
        else:
            raise ValueError(
                f"Неверная конфигурация перехода из '{transition.from_state}': "
                f"нужно указать to_state или routes"
            )
    
    def visualize(self) -> str:
        """Возвращает текстовое представление графа.
        
        Returns:
            Строка с описанием состояний и переходов
        """
        lines = ["📊 Граф агента:", ""]
        
        # Состояния
        lines.append("Состояния:")
        for state in self.states:
            entry_marker = " (entry)" if state.name == self.entry_point else ""
            lines.append(f"  • {state.name}{entry_marker}")
            lines.append(f"    Инструменты: {', '.join(state.tools)}")
            if state.description:
                lines.append(f"    Описание: {state.description}")
        
        lines.append("")
        
        # Переходы
        lines.append("Переходы:")
        for trans in self.transitions:
            if trans.routes:
                routes_str = ", ".join(f"{k}→{v}" for k, v in trans.routes.items())
                lines.append(f"  • {trans.from_state} → [{routes_str}]")
            elif trans.condition:
                lines.append(f"  • {trans.from_state} → {trans.to_state} (условный)")
            else:
                lines.append(f"  • {trans.from_state} → {trans.to_state}")
            
            if trans.description:
                lines.append(f"    {trans.description}")
        
        return "\n".join(lines)
