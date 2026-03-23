"""Общие инструменты и автозагрузка tool-модулей."""
from importlib import import_module
from langchain.tools import tool
from typing import Any, Dict, List, Optional
from pathlib import Path
import pkgutil


_memory_store: Dict[str, Any] = {}
_memory_log: list[str] = []


@tool
def calculator(expression: str) -> str:
    """Вычисляет математическое выражение, например: '2 + 3 * 4'"""
    try:
        allowed_names = {"__builtins__": {}}
        result = eval(expression, allowed_names, {})
        output = str(result)
        memory.invoke(
            {
                "action": "save",
                "key": "calculator_last",
                "value": f"[calculator] {expression} = {output}",
            }
        )
        return output
    except Exception as e:
        output = f"Ошибка вычисления: {e}"
        memory.invoke(
            {
                "action": "save",
                "key": "calculator_last",
                "value": f"[calculator] {expression} -> {output}",
            }
        )
        return output


@tool
def ask_human(question: str) -> str:
    """Задает уточняющий вопрос пользователю и ждет ответа.

    Args:
        question: Вопрос для пользователя

    Returns:
        Ответ пользователя
    """
    print(f"\n🤔 Вопрос агента: {question}", flush=True)
    response = input("👤 Ваш ответ: ")
    return response


@tool
def memory(
    action: str,
    key: str = "",
    value: str = "",
    keys: Optional[List[str]] = None,
    values: Optional[List[str]] = None,
) -> Dict[str, Any]:
    """Сохраняет или читает заметки из памяти агента.

    Пакетно: передай keys и values одинаковой длины (save) или только keys (get) —
    один вызов инструмента вместо нескольких.

    Args:
        action: "save" для сохранения, "get" для чтения, "list" для списка всех ключей
        key: Один ключ (совместимость со старым вызовом)
        value: Одно значение для save (вместе с key)
        keys: Несколько ключей сразу
        values: Значения для save; длина должна совпадать с keys

    Returns:
        Словарь с полями ok, saved/entries/keys или error
    """
    global _memory_store

    def _resolve_key_list() -> List[str]:
        if keys:
            return list(keys)
        if key:
            return [key]
        return []

    if action == "save":
        ks = _resolve_key_list()
        if not ks:
            return {"ok": False, "error": "Нужен key или непустой keys для сохранения"}

        if keys and len(keys) > 0:
            if values is None or len(values) != len(keys):
                return {
                    "ok": False,
                    "error": "Для save с keys нужен values той же длины, что и keys",
                }
            for k, v in zip(keys, values):
                _memory_store[k] = v
            return {"ok": True, "saved": dict(zip(keys, values))}

        _memory_store[key] = value
        return {"ok": True, "saved": {key: value}}

    if action == "get":
        ks = _resolve_key_list()
        if not ks:
            return {"ok": False, "error": "Нужен key или непустой keys для чтения"}
        entries = {k: _memory_store[k] if k in _memory_store else None for k in ks}
        return {"ok": True, "entries": entries}

    if action == "list":
        return {
            "ok": True,
            "keys": list(_memory_store.keys()),
            "empty": len(_memory_store) == 0,
        }

    return {
        "ok": False,
        "error": f"Неизвестное действие: {action}. Используйте 'save', 'get' или 'list'",
    }


def reset_memory() -> None:
    """Сбрасывает память агента (глобальные хранилища)."""
    global _memory_store, _memory_log
    _memory_store = {}
    _memory_log = []


@tool
def memory_append(text: str) -> str:
    """Добавляет строку в журнал памяти (append-only)."""
    global _memory_log
    if text is None:
        return "Ошибка: text не может быть пустым"
    _memory_log.append(str(text))
    return f"✓ Добавлено в журнал: {text}"


@tool
def memory_read(limit: int = 0) -> str:
    """Возвращает журнал памяти. Если limit > 0, возвращает последние N строк."""
    global _memory_log
    if not _memory_log:
        return "Журнал памяти пуст"
    if isinstance(limit, int) and limit > 0:
        lines = _memory_log[-limit:]
    else:
        lines = _memory_log
    return "\n".join(lines)


@tool
def summarize(focus: str = "general") -> str:
    """Создает саммари выполненной работы на основе памяти агента.

    Args:
        focus: Фокус саммари - "general" (общий), "results" (результаты), "process" (процесс)

    Returns:
        Краткое саммари
    """
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

    else:
        summary_parts.append("\n📊 Общая информация:")
        summary_parts.append(f"  - Всего записей: {len(_memory_store)}")
        for key, value in _memory_store.items():
            summary_parts.append(f"  - {key}: {value}")

    return "\n".join(summary_parts)


@tool
def think(thought: str) -> str:
    """Инструмент для внутренних размышлений агента.

    Args:
        thought: Мысль или размышление агента

    Returns:
        Подтверждение размышления
    """
    print(f"\n💭 Размышление агента: {thought}", flush=True)
    return f"✓ Размышление зафиксировано: {thought}"


core_tools = [
    calculator,
    ask_human,
    memory,
    memory_append,
    memory_read,
    summarize,
    think
]


def _load_tools_from_module(module) -> list:
    tools_in_module = getattr(module, "TOOLS", None)
    if tools_in_module is None:
        return []
    if not isinstance(tools_in_module, list):
        raise TypeError(f"В модуле {module.__name__} атрибут TOOLS должен быть list")
    return tools_in_module


def _discover_shared_tools() -> list:
    discovered = []
    package_dir = Path(__file__).resolve().parent
    for module_info in pkgutil.iter_modules([str(package_dir)]):
        module_name = module_info.name
        if module_name in {"tools"} or module_name.startswith("_"):
            continue
        module = import_module(f"src.tools.{module_name}")
        discovered.extend(_load_tools_from_module(module))
    return discovered


def _discover_agent_tools(agent_name: str) -> list:
    if not agent_name:
        return []
    target_module = f"src.agents.{agent_name}.tools"
    try:
        module = import_module(target_module)
    except ModuleNotFoundError as exc:
        if exc.name == target_module:
            return []
        raise
    return _load_tools_from_module(module)


def _discover_all_agent_tools() -> list:
    discovered = []
    agents_dir = Path(__file__).resolve().parent.parent / "agents"
    for module_info in pkgutil.iter_modules([str(agents_dir)]):
        if not module_info.ispkg:
            continue
        name = module_info.name
        if name.startswith("_"):
            continue
        discovered.extend(_discover_agent_tools(name))
    return discovered


def _validate_unique_tool_names(tools_list: list) -> list:
    names = {}
    for tool_obj in tools_list:
        tool_name = getattr(tool_obj, "name", None)
        if not tool_name:
            raise ValueError(f"Найден объект без имени инструмента: {tool_obj}")
        if tool_name in names:
            raise ValueError(f"Дублирующееся имя инструмента: {tool_name}")
        names[tool_name] = True
    return tools_list


def get_tools(agent_name: str | None = None) -> list:
    if agent_name is not None and not isinstance(agent_name, str):
        raise TypeError("agent_name должен быть строкой или None")

    resolved = []
    resolved.extend(core_tools)
    resolved.extend(_discover_shared_tools())
    if agent_name:
        resolved.extend(_discover_agent_tools(agent_name))
    else:
        resolved.extend(_discover_all_agent_tools())
    resolved.extend(multiagent_tools)
    return _validate_unique_tool_names(resolved)


def get_tools_dict(agent_name: str | None = None) -> dict:
    tools_list = get_tools(agent_name=agent_name)
    return {tool.name: tool for tool in tools_list}


# Реестр агентов для мультиагентных систем
_agents_registry: dict = {}


def register_agent(name: str, agent) -> None:
    _agents_registry[name] = agent


def get_registered_agent(name: str):
    return _agents_registry.get(name)


def list_registered_agents() -> list[str]:
    return list(_agents_registry.keys())

@tool
def call_agent(agent_name: str, query: str) -> str:
    """Вызывает другого зарегистрированного агента.

    Args:
        agent_name: Имя агента для вызова
        query: Запрос к агенту

    Returns:
        Результат работы агента
    """
    from src.agent_engine.logging_utils import log_memory_snapshot, is_enabled

    if is_enabled():
        log_memory_snapshot(f"before_call({agent_name})", _memory_store, when="multiagent")

    agent = get_registered_agent(agent_name)
    if not agent:
        available = ", ".join(list_registered_agents()) or "нет"
        return f"Ошибка: агент '{agent_name}' не зарегистрирован. Доступны: {available}"

    try:
        result = agent.invoke([query])
        last_message = result["messages"][-1].content
        return f"Результат от агента '{agent_name}':\n{last_message}"
    except Exception as e:
        return f"Ошибка при вызове агента '{agent_name}': {e}"


multiagent_tools = [
    call_agent
]


tools = get_tools()
