"""Агент аудита файлов проверки."""

from src.agent_engine import AgentConfig, State


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
5. Когда папка подтверждена, используй list_case_files и сохрани в память ВСЕ шесть ключей
   ОДНИМ вызовом memory (пакетно), без нескольких подряд вызовов memory(save) на отдельные ключи:
   memory(
     action="save",
     keys=["case_id", "case_folder", "case_files", "docx_files", "sql_files", "py_files"],
     values=[<номер проверки>, <путь к папке кейса>, <полный список файлов>, <только .docx>, <только .sql>, <только .py>]
   )
   Списки файлов в values можно передать одной строкой (например имена через запятую или перевод строки).

Режимы работы:
- Первый вход: если нет case_id/case_folder/case_files.
- Повторный вход: сначала проверь memory(action="list") и дочитай недостающие ключи одним вызовом
  memory(action="get", keys=[...]) либо по одному key=, затем добери только отсутствующие данные.

Критерий завершения состояния start_work:
- Переходи дальше только когда в памяти сохранены ВСЕ ключи:
  case_id, case_folder, case_files, docx_files, sql_files, py_files.
- Перед transition обязательно проверь их наличие через memory(action="list") и, при необходимости,
  memory(action="get", keys=["case_id", "case_folder", ...]) одним вызовом.

Анти-зацикливание:
- Не задавай один и тот же вопрос пользователю повторно, если case_id уже известен.
- Если find_case_folder вернул status="ok", не вызывай ask_human повторно без новой причины.
- Если не все ключи собраны — делай next_state="stay" с четким списком, каких ключей не хватает.
""",
            # transitions=["analize_word"],
            transitions=["analize_sql"],
            description="Поиск папки проверки и сбор списка файлов",
            memory_injections=[
                ("case_id", "Проверка уже определена: "),
                ("case_folder", "Папка проверки уже найдена: "),
                ("case_files", "Файлы проверки уже собраны: "),
                ("docx_files", "DOCX файлы уже определены: "),
                ("sql_files", "SQL файлы уже определены: "),
                ("py_files", "Python файлы уже определены: "),
            ],
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
1.1 Если sql_files отсутствует:
    - попробуй восстановить его из case_files (выбери только .sql) и сохрани в память как sql_files;
    - если case_files тоже нет, вернись в start_work.
2. Для каждого файла используй read_sql_file.
3. Оцени:
   - токсичность (опасные операции, удаление, эскалация прав)
   - потенциальные проблемы (дропы, отсутствие фильтров, полные сканы)
4. Сохрани в память:
   - sql_verdicts (структура: имя_файла -> анализ и вердикт)

Критерий завершения состояния analize_sql:
- Если sql_verdicts успешно заполнен — переходи в analize_py.
- Если не хватает входных данных и восстановить их не удалось — переходи в start_work.
- Не используй stay более одного раза подряд при одной и той же причине.
""",
            transitions=["analize_py", "start_work"],
            description="Анализ SQL скриптов на риски и проблемы",
            memory_injections=[
                ("case_files", "Из памяти доступны файлы проверки: ", "Список case_files пока отсутствует."),
                ("sql_files", "SQL для анализа уже определены: ", "SQL анализ еще не может стартовать: sql_files отсутствует."),
                ("sql_verdicts", "Анализ SQL уже сделан: "),
            ],
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
            memory_injections=[
                ("py_files", "Python файлы для анализа: ", "Список py_files пока отсутствует."),
                ("py_verdicts", "Анализ Python уже сделан: "),
            ],
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
            memory_injections=[
                ("case_files", "Полный список файлов кейса: "),
                ("docx_files", "DOCX файлы: "),
                ("sql_files", "SQL файлы: "),
                ("py_files", "Python файлы: "),
                ("sql_verdicts", "SQL вердикты: ", "SQL вердикты пока не собраны."),
                ("py_verdicts", "Python вердикты: ", "Python вердикты пока не собраны."),
            ],
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
            memory_injections=[
                ("docx_files", "Отчетный DOCX: "),
                ("sql_verdicts", "Выводы по SQL: ", "Выводы по SQL пока отсутствуют."),
                ("py_verdicts", "Выводы по Python: ", "Выводы по Python пока отсутствуют."),
                ("final_report", "Текущий вариант final_report: "),
            ],
        )
    ]
