"""Классы для декларативного описания состояний агента."""
from dataclasses import dataclass, field
from typing import Callable, Any


@dataclass
class MemoryInjection:
    """Правило инъекции значения из memory в историю сообщений.

    Атрибуты:
        key: Ключ в memory.
        if_exists: Префикс/текст, добавляемый перед значением, если ключ есть.
        if_missing: Текст, добавляемый в историю, если ключ отсутствует.
    """

    key: str
    if_exists: str | None = None
    if_missing: str | None = None


@dataclass
class State:
    """Декларативное описание состояния агента.

    Примеры:
        State(
            name="work",
            tools=["calculator", "ask_human"],
            prompt="Ты помощник для вычислений...",
            transitions=["summarize", "clarify"],
            description="Основное рабочее состояние"
        )

    Атрибуты:
        name: Уникальное имя состояния (используется в графе)
        tools: Список имен инструментов, доступных в этом состоянии
        prompt: Системный промпт для LLM в этом состоянии
        transitions: Список имён состояний, в которые разрешён переход (+ "END")
        description: Описание состояния для документации
        memory_injections: Правила инъекций memory в историю.
            Поддерживаемые форматы элемента:
            - "key"
            - ("key", "if_exists")
            - ("key", "if_exists", "if_missing")
            - {"key": "...", "if_exists": "...", "if_missing": "..."}
            - MemoryInjection(...)
        on_enter: Функция, вызываемая при входе в состояние (опционально)
        on_exit: Функция, вызываемая при выходе из состояния (опционально)
    """
    name: str
    tools: list[str]
    prompt: str
    transitions: list[str] = field(default_factory=list)
    description: str = ""
    memory_injections: list[Any] = field(default_factory=list)
    on_enter: Callable[[dict], dict] | None = None
    on_exit: Callable[[dict], dict] | None = None
