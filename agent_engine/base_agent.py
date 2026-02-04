"""Базовый класс для конфигурации агентов."""
from typing import Any, Dict
from .graph_builder import AgentGraphBuilder
from .state import State, Transition


class AgentConfig:
    """Базовый класс для конфигурации агента.
    
    Наследуйтесь от этого класса и определите:
    - states: список состояний State
    - transitions: список переходов Transition  
    - entry_point: имя начального состояния
    
    Пример:
        class MyAgent(AgentConfig):
            entry_point = "work"
            
            states = [
                State(name="work", tools=["calculator"], prompt="..."),
                State(name="summarize", tools=["summarize"], prompt="...")
            ]
            
            transitions = [
                Transition(from_state="work", to_state="summarize", ...),
                Transition(from_state="summarize", to_state="END", ...)
            ]
        
        # Использование
        agent = MyAgent(llm, tools_dict)
        result = agent.invoke(["Посчитай 2+2"])
        print(result['messages'][-1].content)
    """
    
    # Переопределяются в подклассах
    states: list[State] = []
    transitions: list[Transition] = []
    entry_point: str | None = None
    
    def __init__(self, llm, tools_dict: Dict[str, Any], agent_id: str | None = None):
        """Инициализирует агента.
        
        Args:
            llm: Языковая модель (ChatOpenAI, GigaChat и т.д.)
            tools_dict: Словарь {имя_инструмента: объект_инструмента}
            agent_id: Уникальный идентификатор агента (опционально)
        """
        self.llm = llm
        self.tools_dict = tools_dict
        self.agent_id = agent_id or f"{self.__class__.__name__}_{id(self)}"
        self._graph = None
        
        # Валидация конфигурации
        if not self.states:
            raise ValueError(
                f"{self.__class__.__name__}: Не определены состояния. "
                f"Установите атрибут класса 'states'"
            )
        if not self.entry_point:
            raise ValueError(
                f"{self.__class__.__name__}: Не определена точка входа. "
                f"Установите атрибут класса 'entry_point'"
            )
    
    def build(self):
        """Собирает и возвращает скомпилированный граф агента.
        
        Returns:
            Скомпилированный граф LangGraph
        """
        builder = AgentGraphBuilder(self.llm, self.tools_dict)
        builder.add_states(self.states)
        builder.add_transitions(self.transitions)
        builder.set_entry(self.entry_point)
        
        self._graph = builder.build()
        return self._graph
    
    @property
    def graph(self):
        """Возвращает граф (собирает если еще не собран).
        
        Returns:
            Скомпилированный граф LangGraph
        """
        if self._graph is None:
            self.build()
        return self._graph
    
    def invoke(self, messages: list[str] | dict, config: dict | None = None) -> dict:
        """Запускает агента с сообщениями.
        
        Args:
            messages: Список текстовых сообщений или готовый state dict
            config: Конфигурация для LangGraph (опционально)
            
        Returns:
            Финальное состояние агента с ключами 'messages' и 'memory'
            
        Примеры:
            # Простой запуск
            result = agent.invoke(["Посчитай 2+2"])
            
            # С готовым state
            result = agent.invoke({
                'messages': ["Привет"],
                'memory': {'key': 'value'}
            })
        """
        # Преобразуем входные данные в state
        if isinstance(messages, dict):
            state = messages
        else:
            state = {'messages': messages, 'memory': {}}
        
        return self.graph.invoke(state, config=config)
    
    def stream(self, messages: list[str] | dict, config: dict | None = None):
        """Стримит выполнение агента (для отслеживания промежуточных шагов).
        
        Args:
            messages: Список текстовых сообщений или готовый state dict
            config: Конфигурация для LangGraph (опционально)
            
        Yields:
            Промежуточные состояния агента
            
        Пример:
            for chunk in agent.stream(["Посчитай 2+2"]):
                print(chunk)
        """
        # Преобразуем входные данные в state
        if isinstance(messages, dict):
            state = messages
        else:
            state = {'messages': messages, 'memory': {}}
        
        return self.graph.stream(state, config=config)
    
    def visualize(self) -> str:
        """Возвращает текстовое описание графа агента.
        
        Returns:
            Строка с описанием состояний и переходов
        """
        builder = AgentGraphBuilder(self.llm, self.tools_dict)
        builder.add_states(self.states)
        builder.add_transitions(self.transitions)
        builder.set_entry(self.entry_point)
        
        return builder.visualize()
    
    def __repr__(self) -> str:
        """Строковое представление агента."""
        return f"{self.__class__.__name__}(id={self.agent_id}, states={len(self.states)})"
