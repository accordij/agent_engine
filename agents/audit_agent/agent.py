"""Агент аудита файлов проверки.

Сложный агент с множеством состояний и роутером для демонстрации:
- Многошаговый workflow
- Роутинг с самопроверкой
- Работа с файловой системой
- Анализ разных типов файлов
"""

from agent_engine import AgentConfig, State, Transition, Conditions


def _route_after_self_check(state: dict) -> str:
    """Роутер после самопроверки: выбирает следующее состояние.
    
    Args:
        state: Состояние агента
        
    Returns:
        Имя следующего состояния
    """
    memory = state.get("memory", {})
    next_state = memory.get("next_state", "").strip()
    
    # Проверяем что next_state валидный
    valid_states = {"analize_word", "analize_sql", "analize_py", "write_report"}
    if next_state in valid_states:
        return next_state
    
    # По умолчанию возвращаемся в self_check
    return "self_check"


class AuditAgent(AgentConfig):
    """Агент аудита файлов проверки.
    
    Граф: [start_work] → [analize_word] → [analize_sql] → [analize_py] 
          → [self_check] ⟳ → [write_report] → END
    
    Демонстрирует:
    - Сложный многошаговый workflow (6 состояний)
    - Роутер с самопроверкой и возвратом
    - Работа с файловой системой
    - Анализ документов, SQL, Python
    - Итоговая отчетность
    """
    
    entry_point = "start_work"
    
    states = [
        State(
            name="start_work",
            tools=[
                "ask_human",
                "think",
                "memory",
                "list_data_folders",
                "find_case_folder",
                "list_case_files"
            ],
            prompt="""Ты агент аудита файлов проверки.
Твоя цель на этом шаге: найти папку проверки и собрать список файлов.

Алгоритм:
1. Попроси у пользователя номер проверки, если у тебя его нет.
2. Используй инструмент list_data_folders, чтобы увидеть доступные папки.
3. Используй find_case_folder, чтобы сопоставить ввод пользователя с папками.
4. Если статус "needs_confirmation" или есть сомнения — задай уточняющий вопрос.
5. Когда папка подтверждена, используй list_case_files и выпиши файлы в память:
   - case_id
   - case_folder
   - case_files (полный список)
   - docx_files (только .docx)
   - sql_files (только .sql)
   - py_files (только .py)

Если информации не хватает или она противоречивая, используй think или ask_human.

Когда папка и файлы собраны, скажи ключевую фразу: START_WORK_DONE
""",
            description="Поиск папки проверки и сбор списка файлов"
        ),
        
        State(
            name="analize_word",
            tools=["read_docx_structure", "memory", "think"],
            prompt="""Ты анализируешь docx файлы проверки.
Алгоритм:
1. Прочитай docx_files из памяти.
2. Для каждого файла вызови read_docx_structure.
3. В память сохрани:
   - docx_notes (структура: имя_файла -> найденные реплики/таблицы)
4. Если файлов нет, зафиксируй это в памяти.

Когда все docx обработаны, скажи ключевую фразу: ANALIZE_WORD_DONE
""",
            description="Анализ docx файлов: реплики и таблицы"
        ),
        
        State(
            name="analize_sql",
            tools=["read_sql_file", "memory", "think"],
            prompt="""Ты анализируешь SQL скрипты проверки.
Алгоритм:
1. Прочитай sql_files из памяти.
2. Для каждого файла используй read_sql_file.
3. Оцени:
   - токсичность (опасные операции, удаление, эскалация прав)
   - потенциальные проблемы (дропы, отсутствие фильтров, полные сканы)
4. Сохрани в память:
   - sql_verdicts (структура: имя_файла -> анализ и вердикт)

Когда все SQL обработаны, скажи ключевую фразу: ANALIZE_SQL_DONE
""",
            description="Анализ SQL скриптов на риски и проблемы"
        ),
        
        State(
            name="analize_py",
            tools=["read_py_file", "memory", "think"],
            prompt="""Ты анализируешь Python скрипты проверки.
Алгоритм:
1. Прочитай py_files из памяти.
2. Для каждого файла используй read_py_file.
3. Оцени:
   - токсичность (опасные операции, удаления, сетевые вызовы без контроля)
   - потенциальные проблемы (ошибки, отсутствие проверок, неподписанные источники)
4. Сохрани в память:
   - py_verdicts (структура: имя_файла -> анализ и вердикт)

Когда все Python обработаны, скажи ключевую фразу: ANALIZE_PY_DONE
""",
            description="Анализ Python скриптов на риски и проблемы"
        ),
        
        State(
            name="self_check",
            tools=["memory", "think"],
            prompt="""Ты выполняешь самопроверку полноты анализа.
Алгоритм:
1. Сверь case_files со списками docx/sql/py и вердиктами в памяти.
2. Если чего-то не хватает — выбери следующее состояние:
   - analize_word
   - analize_sql
   - analize_py
3. Если все готово — выбери write_report.
4. Сохрани в память ключ next_state со значением выбранного состояния.

Когда выбор сделан, скажи ключевую фразу: SELF_CHECK_DONE
""",
            description="Проверка полноты обработки и выбор следующего шага"
        ),
        
        State(
            name="write_report",
            tools=["memory", "summarize"],
            prompt="""Ты пишешь итоговый отчет на основе памяти.
Алгоритм:
1. Прочитай memory и собери итог:
   - что было в docx
   - какие выводы по SQL
   - какие выводы по Python
2. Сделай краткий, структурированный отчет.
3. Сохрани отчет в память как final_report.
4. В конце скажи ключевую фразу: REPORT_DONE
""",
            description="Формирование итогового отчета"
        )
    ]
    
    transitions = [
        # Переход 1: start_work → analize_word
        Transition(
            from_state="start_work",
            to_state="analize_word",
            condition=Conditions.contains_keyword("START_WORK_DONE", case_sensitive=False),
            description="Переход после успешного выбора папки и сбора файлов"
        ),
        
        # Переход 2: analize_word → analize_sql
        Transition(
            from_state="analize_word",
            to_state="analize_sql",
            condition=Conditions.contains_keyword("ANALIZE_WORD_DONE", case_sensitive=False),
            description="Переход после анализа docx"
        ),
        
        # Переход 3: analize_sql → analize_py
        Transition(
            from_state="analize_sql",
            to_state="analize_py",
            condition=Conditions.contains_keyword("ANALIZE_SQL_DONE", case_sensitive=False),
            description="Переход после анализа SQL"
        ),
        
        # Переход 4: analize_py → self_check
        Transition(
            from_state="analize_py",
            to_state="self_check",
            condition=Conditions.contains_keyword("ANALIZE_PY_DONE", case_sensitive=False),
            description="Переход после анализа Python"
        ),
        
        # Переход 5: self_check → [роутер]
        Transition(
            from_state="self_check",
            condition=_route_after_self_check,
            routes={
                "analize_word": "analize_word",
                "analize_sql": "analize_sql",
                "analize_py": "analize_py",
                "write_report": "write_report",
                "self_check": "self_check"
            },
            description="Роутер после самопроверки: возврат к анализу или переход к отчету"
        ),
        
        # Переход 6: write_report → END
        Transition(
            from_state="write_report",
            to_state="END",
            condition=Conditions.always_true,
            description="Завершение после формирования отчета"
        )
    ]
