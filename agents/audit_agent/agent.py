"""Агент аудита файлов проверки."""

from agent_engine import AgentConfig, State


class AuditAgent(AgentConfig):
    """Граф: [start_work] → [analize_word] → [analize_sql] → [analize_py]
          → [self_check] ⟳ → [write_report] → END
    """
    
    entry_point = "start_work"
    
    states = [
        State(
            name="start_work",
            tools=["ask_human", "think", "memory", "list_data_folders",
                   "find_case_folder", "list_case_files"],
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
""",
            # transitions=["analize_word"],
            transitions=["analize_sql"],
            description="Поиск папки проверки и сбор списка файлов",
        ),
        
#         State(
#             name="analize_word",
#             tools=["read_docx_structure", "memory", "think"],
#             prompt="""Ты анализируешь docx файлы проверки.
# Алгоритм:
# 1. Прочитай docx_files из памяти.
# 2. Для каждого файла вызови read_docx_structure.
# 3. В память сохрани:
#    - docx_notes (структура: имя_файла -> найденные реплики/таблицы)
# 4. Если файлов нет, зафиксируй это в памяти.
# """,
#             transitions=["analize_sql"],
#             description="Анализ docx файлов: реплики и таблицы",
#         ),
        
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
""",
            transitions=["analize_py"],
            description="Анализ SQL скриптов на риски и проблемы",
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
""",
            transitions=["self_check"],
            description="Анализ Python скриптов на риски и проблемы",
        ),
        
        State(
            name="self_check",
            tools=["memory", "think"],
            prompt="""Ты выполняешь самопроверку полноты анализа.
Алгоритм:
1. Сверь case_files со списками docx/sql/py и вердиктами в памяти.
2. Если чего-то не хватает — перейди в нужное состояние анализа.
3. Если все готово — перейди в write_report.
""",
            transitions=["analize_word", "analize_sql", "analize_py", "write_report"],
            description="Проверка полноты обработки и выбор следующего шага",
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
""",
            transitions=["END"],
            description="Формирование итогового отчета",
        )
    ]
