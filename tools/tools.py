"""Инструменты для агента."""
from langchain.tools import tool
from typing import Dict, Any
from agent_engine.debug import log_prompts_enabled


# Глобальное хранилище памяти для агента
_memory_store: Dict[str, Any] = {}


def _log_tool_call(tool_name: str) -> None:
    if log_prompts_enabled():
        print(f"[TOOL] {tool_name}")


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
    print(f"\n🤔 Вопрос агента: {question}")
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
    print(f"\n💭 Размышление агента: {thought}")
    return f"✓ Размышление зафиксировано: {thought}"


# Экспорт списка инструментов
tools = [calculator, ask_human, memory, summarize, think]
