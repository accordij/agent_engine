"""Простейший тестовый агент с одним состоянием.

Используется для проверки базовой функциональности:
- Создание агента через AgentConfig
- Работа с инструментами
- Переход в END
"""

from agent_engine import AgentConfig, State, Transition, Conditions


class TestAgent(AgentConfig):
    """Простейший агент для тестирования базовой функциональности.
    
    Граф: [work] → END
    
    Используется для проверки:
    - Базовая работа с AgentConfig
    - Вызов инструментов (calculator, memory, think)
    - Безусловный переход в END
    """
    
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
Когда закончишь, скажи "ГОТОВО".
""",
            description="Единственное рабочее состояние"
        )
    ]
    
    transitions = [
        Transition(
            from_state="work",
            to_state="END",
            condition=Conditions.contains_keyword("ГОТОВО", case_sensitive=False),
            description="Завершение работы по ключевому слову"
        )
    ]
