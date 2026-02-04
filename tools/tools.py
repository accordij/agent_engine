"""Инструменты для агента."""
from langchain.tools import tool
from typing import Dict, Any
from agent_engine.debug import log_prompts_enabled, log_event
import os
import re
import json
import difflib
import zipfile
import xml.etree.ElementTree as ET


# Глобальное хранилище памяти для агента
_memory_store: Dict[str, Any] = {}
_memory_log: list[str] = []


def _log_tool_call(tool_name: str, params: dict | None = None) -> None:
    if log_prompts_enabled():
        if params:
            params_str = json.dumps(params, ensure_ascii=False)
            print(f"[TOOL] {tool_name} params={params_str}", flush=True)
        else:
            print(f"[TOOL] {tool_name}", flush=True)
    log_event(
        "tool_call",
        {
            "tool": tool_name,
            "params": params or {},
        },
    )


def _log_tool_result(tool_name: str, result) -> None:
    log_event(
        "tool_result",
        {
            "tool": tool_name,
            "result": result,
        },
    )


@tool
def calculator(expression: str) -> str:
    """Вычисляет математическое выражение, например: '2 + 3 * 4'"""
    _log_tool_call("calculator", {"expression": expression})
    try:
        allowed_names = {"__builtins__": {}}
        result = eval(expression, allowed_names, {})
        output = str(result)
        memory_append.invoke(f"[calculator] {expression} = {output}")
        _log_tool_result("calculator", output)
        return output
    except Exception as e:
        output = f"Ошибка вычисления: {e}"
        memory_append.invoke(f"[calculator] {expression} -> {output}")
        _log_tool_result("calculator", output)
        return output


@tool
def ask_human(question: str) -> str:
    """Задает уточняющий вопрос пользователю и ждет ответа.
    
    Args:
        question: Вопрос для пользователя
        
    Returns:
        Ответ пользователя
    """
    _log_tool_call("ask_human", {"question": question})
    print(f"\n🤔 Вопрос агента: {question}", flush=True)
    response = input("👤 Ваш ответ: ")
    _log_tool_result("ask_human", response)
    return response


@tool
def memory(action: str, key: str = "", value: str = "") -> str:
    """Сохраняет или читает заметки из памяти агента.
    
    Args:
        action: "save" для сохранения, "get" для чтения, "list" для списка всех ключей
        key: Ключ для сохранения/чтения
        value: Значение для сохранения (только для action="save")
        
    Returns:
        Результат операции
    """
    _log_tool_call("memory", {"action": action, "key": key, "value": value})
    global _memory_store
    
    if action == "save":
        if not key:
            output = "Ошибка: нужно указать ключ для сохранения"
            _log_tool_result("memory", output)
            return output
        _memory_store[key] = value
        output = f"✓ Сохранено в память: {key} = {value}"
        _log_tool_result("memory", output)
        return output
    
    elif action == "get":
        if not key:
            output = "Ошибка: нужно указать ключ для чтения"
            _log_tool_result("memory", output)
            return output
        if key in _memory_store:
            output = f"Из памяти: {key} = {_memory_store[key]}"
        else:
            output = f"Ключ '{key}' не найден в памяти"
        _log_tool_result("memory", output)
        return output
    
    elif action == "list":
        if not _memory_store:
            output = "Память пуста"
            _log_tool_result("memory", output)
            return output
        keys = ", ".join(_memory_store.keys())
        output = f"Ключи в памяти: {keys}"
        _log_tool_result("memory", output)
        return output
    
    else:
        output = f"Неизвестное действие: {action}. Используйте 'save', 'get' или 'list'"
        _log_tool_result("memory", output)
        return output


def reset_memory() -> None:
    """Сбрасывает память агента (глобальные хранилища)."""
    global _memory_store, _memory_log
    _memory_store = {}
    _memory_log = []


@tool
def memory_append(text: str) -> str:
    """Добавляет строку в журнал памяти (append-only)."""
    _log_tool_call("memory_append", {"text": text})
    global _memory_log
    if text is None:
        output = "Ошибка: text не может быть пустым"
        _log_tool_result("memory_append", output)
        return output
    _memory_log.append(str(text))
    output = f"✓ Добавлено в журнал: {text}"
    _log_tool_result("memory_append", output)
    return output


@tool
def memory_read(limit: int = 0) -> str:
    """Возвращает журнал памяти. Если limit > 0, возвращает последние N строк."""
    _log_tool_call("memory_read", {"limit": limit})
    global _memory_log
    if not _memory_log:
        output = "Журнал памяти пуст"
        _log_tool_result("memory_read", output)
        return output
    if isinstance(limit, int) and limit > 0:
        lines = _memory_log[-limit:]
    else:
        lines = _memory_log
    output = "\n".join(lines)
    _log_tool_result("memory_read", output)
    return output


@tool
def summarize(focus: str = "general") -> str:
    """Создает саммари выполненной работы на основе памяти агента.
    
    Args:
        focus: Фокус саммари - "general" (общий), "results" (результаты), "process" (процесс)
        
    Returns:
        Краткое саммари
    """
    _log_tool_call("summarize", {"focus": focus})
    global _memory_store
    
    if not _memory_store:
        output = "📝 Саммари: Память пуста, нет данных для создания саммари."
        _log_tool_result("summarize", output)
        return output
    
    summary_parts = ["📝 Саммари выполненной работы:"]
    
    if focus == "results":
        summary_parts.append("\n🎯 Результаты:")
        for key, value in _memory_store.items():
            summary_parts.append(f"  - {key}: {value}")
    
    elif focus == "process":
        summary_parts.append("\n⚙️ Процесс работы:")
        summary_parts.append(f"  - Сохранено {len(_memory_store)} записей в памяти")
        for key in _memory_store.keys():
            summary_parts.append(f"  - Обработано: {key}")
    
    else:  # general
        summary_parts.append("\n📊 Общая информация:")
        summary_parts.append(f"  - Всего записей: {len(_memory_store)}")
        for key, value in _memory_store.items():
            summary_parts.append(f"  - {key}: {value}")
    
    output = "\n".join(summary_parts)
    _log_tool_result("summarize", output)
    return output


@tool
def think(thought: str) -> str:
    """Инструмент для внутренних размышлений агента.
    Помогает агенту структурировать свои мысли перед принятием решений.
    
    Args:
        thought: Мысль или размышление агента
        
    Returns:
        Подтверждение размышления
    """
    _log_tool_call("think", {"thought": thought})
    print(f"\n💭 Размышление агента: {thought}", flush=True)
    output = f"✓ Размышление зафиксировано: {thought}"
    _log_tool_result("think", output)
    return output


@tool
def list_data_folders(data_root: str = "data") -> str:
    """Возвращает список папок в корневой папке data."""
    _log_tool_call("list_data_folders", {"data_root": data_root})
    if not os.path.isdir(data_root):
        output = f"Ошибка: папка '{data_root}' не найдена"
        _log_tool_result("list_data_folders", output)
        return output
    folders = [
        name for name in os.listdir(data_root)
        if os.path.isdir(os.path.join(data_root, name))
    ]
    if not folders:
        output = "Папка data пуста"
        _log_tool_result("list_data_folders", output)
        return output
    folders.sort()
    output = "\n".join(folders)
    _log_tool_result("list_data_folders", output)
    return output


def _normalize_case_name(value: str) -> str:
    value = value.lower()
    value = re.sub(r"[^a-z0-9]+", "", value)
    return value


def _score_case_match(user_input: str, folder_name: str) -> float:
    normalized_input = _normalize_case_name(user_input)
    normalized_folder = _normalize_case_name(folder_name)
    if not normalized_input or not normalized_folder:
        return 0.0
    ratio = difflib.SequenceMatcher(None, normalized_input, normalized_folder).ratio()
    input_digits = "".join(re.findall(r"\d+", user_input))
    folder_digits = "".join(re.findall(r"\d+", folder_name))
    digit_score = 0.0
    if input_digits and folder_digits:
        if input_digits in folder_digits or folder_digits in input_digits:
            digit_score = 1.0
        else:
            digit_score = difflib.SequenceMatcher(None, input_digits, folder_digits).ratio()
    return 0.7 * ratio + 0.3 * digit_score


@tool
def find_case_folder(case_input: str, data_root: str = "data") -> str:
    """Находит наиболее похожую папку проверки по вводу пользователя.

    Возвращает JSON со статусом и кандидатами:
    - status: "ok" или "needs_confirmation"
    - match: путь к папке (если status="ok")
    - candidates: топ-5 кандидатов с оценками
    """
    _log_tool_call("find_case_folder", {"case_input": case_input, "data_root": data_root})
    if not case_input:
        output = json.dumps({"status": "needs_confirmation", "reason": "empty_input"}, ensure_ascii=False)
        _log_tool_result("find_case_folder", output)
        return output
    if not os.path.isdir(data_root):
        output = json.dumps({"status": "error", "reason": "data_root_not_found"}, ensure_ascii=False)
        _log_tool_result("find_case_folder", output)
        return output

    folders = [
        name for name in os.listdir(data_root)
        if os.path.isdir(os.path.join(data_root, name))
    ]
    if not folders:
        output = json.dumps({"status": "error", "reason": "no_folders"}, ensure_ascii=False)
        _log_tool_result("find_case_folder", output)
        return output

    normalized_input = _normalize_case_name(case_input)
    substring_matches = []
    if normalized_input:
        for folder in folders:
            normalized_folder = _normalize_case_name(folder)
            if normalized_input in normalized_folder:
                match_score = round(len(normalized_input) / max(len(normalized_folder), 1), 3)
                substring_matches.append({"folder": folder, "score": match_score})

    if substring_matches:
        substring_matches.sort(key=lambda x: x["score"], reverse=True)
        if len(substring_matches) == 1:
            best = substring_matches[0]
            output = json.dumps({
                "status": "ok",
                "match": os.path.join(data_root, best["folder"]),
                "score": best["score"],
                "match_mode": "substring",
                "candidates": substring_matches
            }, ensure_ascii=False)
            _log_tool_result("find_case_folder", output)
            return output
        output = json.dumps({
            "status": "needs_confirmation",
            "match_mode": "substring",
            "candidates": substring_matches[:5]
        }, ensure_ascii=False)
        _log_tool_result("find_case_folder", output)
        return output

    scored = []
    for folder in folders:
        score = _score_case_match(case_input, folder)
        scored.append({"folder": folder, "score": round(score, 3)})

    scored.sort(key=lambda x: x["score"], reverse=True)
    top_candidates = scored[:5]

    best = top_candidates[0]
    second = top_candidates[1] if len(top_candidates) > 1 else None
    confident = best["score"] >= 0.75 and (second is None or (best["score"] - second["score"]) >= 0.08)

    if confident:
        output = json.dumps({
            "status": "ok",
            "match": os.path.join(data_root, best["folder"]),
            "score": best["score"],
            "candidates": top_candidates
        }, ensure_ascii=False)
        _log_tool_result("find_case_folder", output)
        return output

    output = json.dumps({
        "status": "needs_confirmation",
        "candidates": top_candidates
    }, ensure_ascii=False)
    _log_tool_result("find_case_folder", output)
    return output


@tool
def list_case_files(case_folder: str) -> str:
    """Возвращает список файлов в папке проверки."""
    _log_tool_call("list_case_files", {"case_folder": case_folder})
    if not case_folder:
        output = "Ошибка: не указан путь к папке"
        _log_tool_result("list_case_files", output)
        return output
    if not os.path.isdir(case_folder):
        output = f"Ошибка: папка '{case_folder}' не найдена"
        _log_tool_result("list_case_files", output)
        return output
    files = [
        name for name in os.listdir(case_folder)
        if os.path.isfile(os.path.join(case_folder, name))
    ]
    if not files:
        output = "В папке нет файлов"
        _log_tool_result("list_case_files", output)
        return output
    files.sort()
    output = "\n".join(files)
    _log_tool_result("list_case_files", output)
    return output


def _extract_docx_text_nodes(xml_content: str) -> list[str]:
    namespace = {"w": "http://schemas.openxmlformats.org/wordprocessingml/2006/main"}
    root = ET.fromstring(xml_content)
    paragraphs = []
    for p in root.findall(".//w:p", namespace):
        texts = [t.text for t in p.findall(".//w:t", namespace) if t.text]
        if texts:
            paragraphs.append("".join(texts).strip())
    return paragraphs


@tool
def read_docx_structure(docx_path: str) -> str:
    """Читает docx и возвращает названия реплик и таблиц (если найдены)."""
    _log_tool_call("read_docx_structure", {"docx_path": docx_path})
    if not docx_path:
        output = json.dumps({"status": "error", "reason": "empty_path"}, ensure_ascii=False)
        _log_tool_result("read_docx_structure", output)
        return output
    if not os.path.isfile(docx_path):
        output = json.dumps({"status": "error", "reason": "file_not_found"}, ensure_ascii=False)
        _log_tool_result("read_docx_structure", output)
        return output

    try:
        with zipfile.ZipFile(docx_path, "r") as archive:
            with archive.open("word/document.xml") as doc_xml:
                xml_content = doc_xml.read().decode("utf-8", errors="ignore")
    except Exception as exc:
        output = json.dumps({"status": "error", "reason": f"docx_read_failed: {exc}"}, ensure_ascii=False)
        _log_tool_result("read_docx_structure", output)
        return output

    paragraphs = _extract_docx_text_nodes(xml_content)
    replica_titles = []
    table_titles = []
    for text in paragraphs:
        if re.search(r"\bреплика\b", text, flags=re.IGNORECASE):
            replica_titles.append(text)
        if re.search(r"\bтаблица\b", text, flags=re.IGNORECASE):
            table_titles.append(text)

    output = json.dumps({
        "status": "ok",
        "replica_titles": replica_titles[:50],
        "table_titles": table_titles[:50],
        "total_paragraphs": len(paragraphs)
    }, ensure_ascii=False)
    _log_tool_result("read_docx_structure", output)
    return output


@tool
def read_sql_file(sql_path: str) -> str:
    """Читает SQL-файл."""
    _log_tool_call("read_sql_file", {"sql_path": sql_path})
    if not sql_path:
        output = "Ошибка: не указан путь к файлу"
        _log_tool_result("read_sql_file", output)
        return output
    if not os.path.isfile(sql_path):
        output = f"Ошибка: файл '{sql_path}' не найден"
        _log_tool_result("read_sql_file", output)
        return output
    with open(sql_path, "r", encoding="utf-8") as file:
        output = file.read()
    _log_tool_result("read_sql_file", output)
    return output


@tool
def read_py_file(py_path: str) -> str:
    """Читает Python-файл."""
    _log_tool_call("read_py_file", {"py_path": py_path})
    if not py_path:
        output = "Ошибка: не указан путь к файлу"
        _log_tool_result("read_py_file", output)
        return output
    if not os.path.isfile(py_path):
        output = f"Ошибка: файл '{py_path}' не найден"
        _log_tool_result("read_py_file", output)
        return output
    with open(py_path, "r", encoding="utf-8") as file:
        output = file.read()
    _log_tool_result("read_py_file", output)
    return output


# Экспорт списка инструментов (базовые)
tools = [
    calculator,
    ask_human,
    memory,
    memory_append,
    memory_read,
    summarize,
    think,
    list_data_folders,
    find_case_folder,
    list_case_files,
    read_docx_structure,
    read_sql_file,
    read_py_file
]

# ============================================================
# ВСПОМОГАТЕЛЬНЫЕ ФУНКЦИИ
# ============================================================

def get_tools_dict(tools_list: list = None) -> dict:
    """Преобразует список инструментов в словарь {имя: объект}.
    
    Args:
        tools_list: Список инструментов (по умолчанию используется tools)
        
    Returns:
        Словарь {имя_инструмента: объект_инструмента}
    """
    if tools_list is None:
        tools_list = tools
    return {tool.name: tool for tool in tools_list}


# ============================================================
# РЕЕСТР АГЕНТОВ ДЛЯ МУЛЬТИАГЕНТНЫХ СИСТЕМ
# ============================================================

_agents_registry: dict = {}


def register_agent(name: str, agent) -> None:
    """Регистрирует агента в глобальном реестре для вызова другими агентами.
    
    Args:
        name: Уникальное имя агента
        agent: Объект агента (должен иметь метод invoke)
    """
    _agents_registry[name] = agent
    if log_prompts_enabled():
        print(f"[REGISTRY] Зарегистрирован агент: {name}")


def get_registered_agent(name: str):
    """Получает зарегистрированного агента по имени.
    
    Args:
        name: Имя агента
        
    Returns:
        Объект агента или None если не найден
    """
    return _agents_registry.get(name)


def list_registered_agents() -> list[str]:
    """Возвращает список имен зарегистрированных агентов."""
    return list(_agents_registry.keys())


@tool
def call_agent(agent_name: str, query: str) -> str:
    """Вызывает другого зарегистрированного агента.
    
    ВАЖНО: Агент должен быть предварительно зарегистрирован через register_agent()
    
    Args:
        agent_name: Имя агента для вызова
        query: Запрос к агенту
        
    Returns:
        Результат работы агента
        
    Пример:
        # Регистрация агента
        register_agent("math_agent", math_agent_instance)
        
        # Вызов из другого агента
        result = call_agent("math_agent", "Посчитай 2+2")
    """
    _log_tool_call("call_agent", {"agent_name": agent_name, "query": query})
    
    agent = get_registered_agent(agent_name)
    if not agent:
        available = ", ".join(list_registered_agents()) or "нет"
        output = f"Ошибка: агент '{agent_name}' не зарегистрирован. Доступны: {available}"
        _log_tool_result("call_agent", output)
        return output
    
    try:
        result = agent.invoke([query])
        # Получаем последнее сообщение от агента
        last_message = result['messages'][-1].content
        output = f"Результат от агента '{agent_name}':\n{last_message}"
        _log_tool_result("call_agent", output)
        return output
    except Exception as e:
        output = f"Ошибка при вызове агента '{agent_name}': {e}"
        _log_tool_result("call_agent", output)
        return output


# Инструменты для мультиагентных систем (добавляются отдельно при необходимости)
multiagent_tools = [
    call_agent
]
