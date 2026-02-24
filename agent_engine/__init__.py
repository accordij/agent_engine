"""Движок для создания агентов с графом состояний.

Переходы реализованы через transition tool — LLM сам решает
когда перейти и вызывает transition(next_state=..., summary=...).

Пример:
    from agent_engine import State, AgentGraphBuilder, AgentConfig
    
    class MyAgent(AgentConfig):
        entry_point = "work"
        states = [
            State(
                name="work",
                tools=["calculator"],
                prompt="Ты помощник...",
                transitions=["summarize"],
            ),
            State(
                name="summarize",
                tools=["memory"],
                prompt="Подведи итоги...",
                transitions=["END"],
            ),
        ]
"""

from .state import State
from .graph_builder import AgentGraphBuilder
from .base_agent import AgentConfig

__all__ = [
    'State',
    'AgentGraphBuilder',
    'AgentConfig',
]
