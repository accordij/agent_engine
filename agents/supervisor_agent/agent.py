"""Агент-супервизор для мультиагентной системы."""

from agent_engine import AgentConfig, State


class SupervisorAgent(AgentConfig):
    """Граф: [delegate] → [aggregate] → END
    
    ВАЖНО: Перед использованием нужно зарегистрировать подчиненных агентов:
        from tools.tools import register_agent
        register_agent("test_agent", test_agent_instance)
    """
    
    entry_point = "delegate"
    
    states = [
        State(
            name="delegate",
            tools=["call_agent", "memory", "think"],
            prompt="""Ты агент-супервизор, который координирует работу других агентов.

Твои возможности:
- call_agent: вызвать другого зарегистрированного агента
- memory: сохранить результаты работы агентов
- think: продумать стратегию делегирования

Алгоритм работы:
1. Проанализируй запрос пользователя
2. Определи какому агенту делегировать задачу
3. Вызови нужного агента через call_agent(agent_name="...", query="...")
4. Сохрани результат в память
5. Если нужно вызвать еще агентов - сделай это
""",
            transitions=["aggregate"],
            description="Делегирование задач специализированным агентам",
        ),
        
        State(
            name="aggregate",
            tools=["memory", "summarize", "think"],
            prompt="""Ты агрегируешь результаты от разных агентов.

Твоя задача:
1. Прочитай из памяти результаты работы агентов
2. Проанализируй полученные данные
3. Создай итоговый связный ответ пользователю
4. Используй summarize для создания структурированного ответа
""",
            transitions=["END"],
            description="Агрегация результатов от агентов",
        )
    ]
