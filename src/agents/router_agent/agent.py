"""Агент с роутингом — выбор пути в зависимости от типа запроса."""

from src.agent_engine import AgentConfig, State, AutoTransitionRule


class RouterAgent(AgentConfig):
    """Граф: [classify] → [math | text | fast_demo | auto_prepare | error] → END"""

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
3. "fast_demo" — тест быстрого перехода (fast + early_break)
4. "auto_demo" — тест автоперехода без вызова LLM (через memory)
5. "error" — не удалось определить тип

Алгоритм:
1. Проанализируй запрос пользователя
2. Определи его тип
3. Сохрани тип в память: memory(action="save", key="request_type", value="<тип>")
""",
            transitions=["math", "text", "fast_demo", "auto_prepare", "error"],
            description="Классификация типа запроса",
            memory_injections=[
                ("request_type", "Тип запроса уже определен: "),
            ],
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
            memory_injections=[
                ("request_type", "Роутер определил тип: ", "Тип запроса пока не определен."),
                ("result", "Предыдущий результат вычислений: "),
            ],
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
            memory_injections=[
                ("request_type", "Роутер определил тип: ", "Тип запроса пока не определен."),
                ("response_text", "Черновик ответа уже есть: "),
            ],
        ),
        State(
            name="fast_demo",
            tools=["think"],
            prompt="""Ты демонстрационный state для fast transition.

Сделай один короткий шаг и сразу вызови transition(next_state="END").
summary/reasoning можно не передавать.
""",
            transitions=["END"],
            description="Демо быстрого перехода fast_transition + early_break",
            fast_transition=True,
            early_break=True,
            require_transition_summary=False,
            require_transition_reasoning=False,
        ),
        State(
            name="auto_prepare",
            tools=["memory", "think"],
            prompt="""Подготовь данные для автоперехода:
1. Сохрани memory(action="save", key="auto_ready", value="yes")
2. Сразу вызови transition(next_state="auto_demo")
""",
            transitions=["auto_demo"],
            description="Подготовка ключа памяти для auto_transition",
        ),
        State(
            name="auto_demo",
            tools=["think"],
            prompt="""Если ты видишь этот prompt, значит авто-переход не сработал.
Сообщи об ошибке и вызови transition(next_state="END").
""",
            transitions=["END"],
            description="Демо auto_transition по ключу памяти",
            auto_transitions=[
                AutoTransitionRule(
                    next_state="END",
                    summary="Автопереход: найден ключ auto_ready в памяти.",
                    memory_has_all=["auto_ready"],
                )
            ],
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
