"""Инструменты для агента."""
from langchain.tools import tool
from typing import Dict, Any
from agent_engine.debug import log_prompts_enabled
import os
import re
import json
import difflib
import zipfile
import xml.etree.ElementTree as ET


# Глобальное хранилище памяти для агента
_memory_store: Dict[str, Any] = {}


def _log_tool_call(tool_name: str) -> None:
    if log_prompts_enabled():
        print(f"[TOOL] {tool_name}", flush=True)


@tool
def calculator(expression: str) -> str:
    """Вычисляет математическое выражение, например: '2 + 3 * 4'"""
    _log_tool_call("calculator")
    try:
        allowed_names = {"__builtins__": {}}
        result = eval(expression, allowed_names, {})
        return str(result)
    except Exception as e:
        return f"Ошибка вычисления: {e}"


@tool
def ask_human(question: str) -> str:
    """Задает уточняющий вопрос пользователю и ждет ответа.
    
    Args:
        question: Вопрос для пользователя
        
    Returns:
        Ответ пользователя
    """
    _log_tool_call("ask_human")
    print(f"\n🤔 Вопрос агента: {question}", flush=True)
    response = input("👤 Ваш ответ: ")
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
    _log_tool_call("memory")
    global _memory_store
    
    if action == "save":
        if not key:
            return "Ошибка: нужно указать ключ для сохранения"
        _memory_store[key] = value
        return f"✓ Сохранено в память: {key} = {value}"
    
    elif action == "get":
        if not key:
            return "Ошибка: нужно указать ключ для чтения"
        if key in _memory_store:
            return f"Из памяти: {key} = {_memory_store[key]}"
        else:
            return f"Ключ '{key}' не найден в памяти"
    
    elif action == "list":
        if not _memory_store:
            return "Память пуста"
        keys = ", ".join(_memory_store.keys())
        return f"Ключи в памяти: {keys}"
    
    else:
        return f"Неизвестное действие: {action}. Используйте 'save', 'get' или 'list'"


@tool
def summarize(focus: str = "general") -> str:
    """Создает саммари выполненной работы на основе памяти агента.
    
    Args:
        focus: Фокус саммари - "general" (общий), "results" (результаты), "process" (процесс)
        
    Returns:
        Краткое саммари
    """
    _log_tool_call("summarize")
    global _memory_store
    
    if not _memory_store:
        return "📝 Саммари: Память пуста, нет данных для создания саммари."
    
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
    
    return "\n".join(summary_parts)


@tool
def think(thought: str) -> str:
    """Инструмент для внутренних размышлений агента.
    Помогает агенту структурировать свои мысли перед принятием решений.
    
    Args:
        thought: Мысль или размышление агента
        
    Returns:
        Подтверждение размышления
    """
    _log_tool_call("think")
    print(f"\n💭 Размышление агента: {thought}", flush=True)
    return f"✓ Размышление зафиксировано: {thought}"


@tool
def list_data_folders(data_root: str = "data") -> str:
    """Возвращает список папок в корневой папке data."""
    _log_tool_call("list_data_folders")
    if not os.path.isdir(data_root):
        return f"Ошибка: папка '{data_root}' не найдена"
    folders = [
        name for name in os.listdir(data_root)
        if os.path.isdir(os.path.join(data_root, name))
    ]
    if not folders:
        return "Папка data пуста"
    folders.sort()
    return "\n".join(folders)


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
    _log_tool_call("find_case_folder")
    if not case_input:
        return json.dumps({"status": "needs_confirmation", "reason": "empty_input"}, ensure_ascii=False)
    if not os.path.isdir(data_root):
        return json.dumps({"status": "error", "reason": "data_root_not_found"}, ensure_ascii=False)

    folders = [
        name for name in os.listdir(data_root)
        if os.path.isdir(os.path.join(data_root, name))
    ]
    if not folders:
        return json.dumps({"status": "error", "reason": "no_folders"}, ensure_ascii=False)

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
            return json.dumps({
                "status": "ok",
                "match": os.path.join(data_root, best["folder"]),
                "score": best["score"],
                "match_mode": "substring",
                "candidates": substring_matches
            }, ensure_ascii=False)
        return json.dumps({
            "status": "needs_confirmation",
            "match_mode": "substring",
            "candidates": substring_matches[:5]
        }, ensure_ascii=False)

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
        return json.dumps({
            "status": "ok",
            "match": os.path.join(data_root, best["folder"]),
            "score": best["score"],
            "candidates": top_candidates
        }, ensure_ascii=False)

    return json.dumps({
        "status": "needs_confirmation",
        "candidates": top_candidates
    }, ensure_ascii=False)


@tool
def list_case_files(case_folder: str) -> str:
    """Возвращает список файлов в папке проверки."""
    _log_tool_call("list_case_files")
    if not case_folder:
        return "Ошибка: не указан путь к папке"
    if not os.path.isdir(case_folder):
        return f"Ошибка: папка '{case_folder}' не найдена"
    files = [
        name for name in os.listdir(case_folder)
        if os.path.isfile(os.path.join(case_folder, name))
    ]
    if not files:
        return "В папке нет файлов"
    files.sort()
    return "\n".join(files)


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
    _log_tool_call("read_docx_structure")
    if not docx_path:
        return json.dumps({"status": "error", "reason": "empty_path"}, ensure_ascii=False)
    if not os.path.isfile(docx_path):
        return json.dumps({"status": "error", "reason": "file_not_found"}, ensure_ascii=False)

    try:
        with zipfile.ZipFile(docx_path, "r") as archive:
            with archive.open("word/document.xml") as doc_xml:
                xml_content = doc_xml.read().decode("utf-8", errors="ignore")
    except Exception as exc:
        return json.dumps({"status": "error", "reason": f"docx_read_failed: {exc}"}, ensure_ascii=False)

    paragraphs = _extract_docx_text_nodes(xml_content)
    replica_titles = []
    table_titles = []
    for text in paragraphs:
        if re.search(r"\bреплика\b", text, flags=re.IGNORECASE):
            replica_titles.append(text)
        if re.search(r"\bтаблица\b", text, flags=re.IGNORECASE):
            table_titles.append(text)

    return json.dumps({
        "status": "ok",
        "replica_titles": replica_titles[:50],
        "table_titles": table_titles[:50],
        "total_paragraphs": len(paragraphs)
    }, ensure_ascii=False)


@tool
def read_sql_file(sql_path: str) -> str:
    """Читает SQL-файл."""
    _log_tool_call("read_sql_file")
    if not sql_path:
        return "Ошибка: не указан путь к файлу"
    if not os.path.isfile(sql_path):
        return f"Ошибка: файл '{sql_path}' не найден"
    with open(sql_path, "r", encoding="utf-8") as file:
        return file.read()


@tool
def read_py_file(py_path: str) -> str:
    """Читает Python-файл."""
    _log_tool_call("read_py_file")
    if not py_path:
        return "Ошибка: не указан путь к файлу"
    if not os.path.isfile(py_path):
        return f"Ошибка: файл '{py_path}' не найден"
    with open(py_path, "r", encoding="utf-8") as file:
        return file.read()


# Экспорт списка инструментов
tools = [
    calculator,
    ask_human,
    memory,
    summarize,
    think,
    list_data_folders,
    find_case_folder,
    list_case_files,
    read_docx_structure,
    read_sql_file,
    read_py_file
]
