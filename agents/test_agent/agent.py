"""Простейший тестовый агент с одним состоянием."""

from agent_engine import AgentConfig, State


class TestAgent(AgentConfig):
    """Граф: [work] → END"""
    
    entry_point = "work"
    
    states = [
        State(
            name="work",
            tools=["calculator", "memory", "think"],
            prompt="""Ты простой тестовый агент.

Твои возможности:
- calculator: вычисление математических выражений
- memory: сохранение/чтение данных
- think: размышления

Выполни запрос пользователя, используя доступные инструменты.
""",
            transitions=["END"],
            description="Единственное рабочее состояние",
        )
    ]
