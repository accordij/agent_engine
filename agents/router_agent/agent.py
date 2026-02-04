"""Агент с роутингом - выбор пути в зависимости от типа запроса.

Демонстрирует условное ветвление графа:
- Классификация запроса
- Выбор пути обработки
- Роутер с несколькими выходами
"""

from agent_engine import AgentConfig, State, Transition, Conditions


def _route_by_type(state: dict) -> str:
    """Роутер: определяет тип запроса и выбирает путь обработки.
    
    Args:
        state: Состояние агента
        
    Returns:
        Имя следующего состояния: "math", "text" или "error"
    """
    memory = state.get("memory", {})
    request_type = memory.get("request_type", "").lower()
    
    if request_type == "math":
        return "math"
    elif request_type == "text":
        return "text"
    else:
        return "error"


class RouterAgent(AgentConfig):
    """Агент с роутингом для демонстрации ветвления графа.
    
    Граф: [classify] → [math | text | error] → END
    
    Демонстрирует:
    - Классификация запроса
    - Условный роутинг (несколько выходов из состояния)
    - Обработка разных типов запросов
    """
    
    entry_point = "classify"
    
    states = [
        State(
            name="classify",
            tools=["think", "memory"],
            prompt="""Ты агент классификации запросов.

Твоя задача: определить тип запроса пользователя.

Типы запросов:
1. "math" - математический запрос (вычисления, формулы, числа)
2. "text" - текстовый запрос (обычный текст, приветствие, вопрос)

Алгоритм:
1. Проанализируй запрос пользователя
2. Определи его тип (math или text)
3. Сохрани тип в память: memory(action="save", key="request_type", value="<тип>")
4. Скажи "КЛАССИФИКАЦИЯ_ЗАВЕРШЕНА"

Примеры:
- "Посчитай 5+5" → math
- "Привет!" → text
- "Сколько будет 2*3?" → math
- "Как дела?" → text
""",
            description="Классификация типа запроса"
        ),
        
        State(
            name="math",
            tools=["calculator", "memory", "think"],
            prompt="""Ты математический агент.

Обрабатывай математический запрос:
1. Используй calculator для вычислений
2. Сохрани результат в память
3. Дай понятный ответ пользователю
4. Скажи "ОБРАБОТКА_ЗАВЕРШЕНА"
""",
            description="Обработка математических запросов"
        ),
        
        State(
            name="text",
            tools=["memory", "think"],
            prompt="""Ты текстовый агент.

Обрабатывай текстовый запрос:
1. Дай вежливый и полезный ответ
2. Сохрани суть ответа в память
3. Скажи "ОБРАБОТКА_ЗАВЕРШЕНА"
""",
            description="Обработка текстовых запросов"
        ),
        
        State(
            name="error",
            tools=["think"],
            prompt="""Ты обработчик ошибок.

Запрос не был классифицирован.
Попроси пользователя уточнить запрос и скажи "ОБРАБОТКА_ЗАВЕРШЕНА"
""",
            description="Обработка нераспознанных запросов"
        )
    ]
    
    transitions = [
        # Переход 1: Classify → [math | text | error] (роутер)
        Transition(
            from_state="classify",
            condition=_route_by_type,
            routes={
                "math": "math",
                "text": "text",
                "error": "error"
            },
            description="Роутинг по типу запроса после классификации"
        ),
        
        # Переходы 2-4: Обработчики → END
        Transition(
            from_state="math",
            to_state="END",
            condition=Conditions.contains_keyword("ОБРАБОТКА_ЗАВЕРШЕНА", case_sensitive=False),
            description="Завершение после обработки математического запроса"
        ),
        
        Transition(
            from_state="text",
            to_state="END",
            condition=Conditions.contains_keyword("ОБРАБОТКА_ЗАВЕРШЕНА", case_sensitive=False),
            description="Завершение после обработки текстового запроса"
        ),
        
        Transition(
            from_state="error",
            to_state="END",
            condition=Conditions.contains_keyword("ОБРАБОТКА_ЗАВЕРШЕНА", case_sensitive=False),
            description="Завершение после обработки ошибки"
        )
    ]
