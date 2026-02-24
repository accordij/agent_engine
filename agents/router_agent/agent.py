"""Агент с роутингом — выбор пути в зависимости от типа запроса."""

from agent_engine import AgentConfig, State


class RouterAgent(AgentConfig):
    """Граф: [classify] → [math | text | error] → END"""
    
    entry_point = "classify"
    
    states = [
        State(
            name="classify",
            tools=["think", "memory"],
            prompt="""Ты агент классификации запросов.

Твоя задача: определить тип запроса пользователя.

Типы запросов:
1. "math" — математический запрос (вычисления, формулы, числа)
2. "text" — текстовый запрос (обычный текст, приветствие, вопрос)
3. "error" — не удалось определить тип

Алгоритм:
1. Проанализируй запрос пользователя
2. Определи его тип
3. Сохрани тип в память: memory(action="save", key="request_type", value="<тип>")
""",
            transitions=["math", "text", "error"],
            description="Классификация типа запроса",
        ),
        
        State(
            name="math",
            tools=["calculator", "memory", "think"],
            prompt="""Ты математический агент.

Обработай математический запрос:
1. найди запрос в памяти, используй memmory()
2. Используй calculator для вычислений
3. Сохрани результат в память
4. Дай понятный ответ пользователю
""",
            transitions=["END"],
            description="Обработка математических запросов",
        ),
        
        State(
            name="text",
            tools=["memory", "think"],
            prompt="""Ты текстовый агент.

Обработай текстовый запрос:
1. найди запрос в памяти, используй memmory()
2. Дай вежливый и полезный ответ
3. Сохрани суть ответа в память
""",
            transitions=["END"],
            description="Обработка текстовых запросов",
        ),
        
        State(
            name="error",
            tools=["think"],
            prompt="""Ты обработчик ошибок.

Запрос не был классифицирован.
Попроси пользователя уточнить запрос.
""",
            transitions=["END"],
            description="Обработка нераспознанных запросов",
        )
    ]
