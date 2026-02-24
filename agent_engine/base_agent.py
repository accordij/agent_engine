"""Базовый класс для конфигурации агентов."""
from typing import Any, Dict
from .graph_builder import AgentGraphBuilder
from .state import State


class AgentConfig:
    """Базовый класс для конфигурации агента.
    
    Наследуйтесь от этого класса и определите:
    - states: список состояний State (с transitions внутри)
    - entry_point: имя начального состояния
    
    Пример:
        class MyAgent(AgentConfig):
            entry_point = "work"
            states = [
                State(name="work", tools=["calculator"], prompt="...",
                      transitions=["summarize"]),
                State(name="summarize", tools=["summarize"], prompt="...",
                      transitions=["END"]),
            ]
        
        agent = MyAgent(llm, tools_dict)
        result = agent.invoke(["Посчитай 2+2"])
    """
    
    states: list[State] = []
    entry_point: str | None = None
    
    def __init__(self, llm, tools_dict: Dict[str, Any], agent_id: str | None = None):
        self.llm = llm
        self.tools_dict = tools_dict
        self.agent_id = agent_id or f"{self.__class__.__name__}_{id(self)}"
        self._graph = None
        
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
        """Собирает и возвращает скомпилированный граф агента."""
        builder = AgentGraphBuilder(self.llm, self.tools_dict)
        builder.add_states(self.states)
        builder.set_entry(self.entry_point)
        
        self._graph = builder.build()
        return self._graph
    
    @property
    def graph(self):
        """Возвращает граф (собирает если ещё не собран)."""
        if self._graph is None:
            self.build()
        return self._graph
    
    def invoke(self, messages: list[str] | dict, config: dict | None = None) -> dict:
        """Запускает агента с сообщениями."""
        if isinstance(messages, dict):
            state = messages
        else:
            state = {'messages': messages, 'memory': {}, 'summary': ''}
        
        default_config = {"recursion_limit": 100}
        if config:
            default_config.update(config)
        
        return self.graph.invoke(state, config=default_config)
    
    def stream(self, messages: list[str] | dict, config: dict | None = None):
        """Стримит выполнение агента."""
        if isinstance(messages, dict):
            state = messages
        else:
            state = {'messages': messages, 'memory': {}, 'summary': ''}
        
        default_config = {"recursion_limit": 100}
        if config:
            default_config.update(config)
        
        return self.graph.stream(state, config=default_config)
    
    def visualize(self) -> str:
        """Возвращает текстовое описание графа агента."""
        builder = AgentGraphBuilder(self.llm, self.tools_dict)
        builder.add_states(self.states)
        builder.set_entry(self.entry_point)
        return builder.visualize()
    
    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(id={self.agent_id}, states={len(self.states)})"
