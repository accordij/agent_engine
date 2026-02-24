"""Классы для декларативного описания состояний агента."""
from dataclasses import dataclass, field
from typing import Callable


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
        on_enter: Функция, вызываемая при входе в состояние (опционально)
        on_exit: Функция, вызываемая при выходе из состояния (опционально)
    """
    name: str
    tools: list[str]
    prompt: str
    transitions: list[str] = field(default_factory=list)
    description: str = ""
    on_enter: Callable[[dict], dict] | None = None
    on_exit: Callable[[dict], dict] | None = None
