"""Движок для создания агентов с графом состояний.

Пример использования:
    from agent_engine import State, Transition, Conditions, AgentGraphBuilder
    
    # Описываем состояния
    work_state = State(
        name="work",
        tools=["calculator", "ask_human"],
        prompt="Ты помощник..."
    )
    
    # Описываем переходы
    transitions = [
        Transition(
            from_state="work",
            to_state="summarize",
            condition=Conditions.contains_keyword("ГОТОВО")
        )
    ]
    
    # Собираем граф
    builder = AgentGraphBuilder(llm, tools_dict)
    graph = (builder
        .add_state(work_state)
        .add_transitions(transitions)
        .set_entry("work")
        .build())
"""

from .state import State, Transition, Conditions
from .graph_builder import AgentGraphBuilder

__all__ = [
    'State',
    'Transition', 
    'Conditions',
    'AgentGraphBuilder'
]
