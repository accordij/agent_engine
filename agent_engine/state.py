"""Классы для декларативного описания состояний и переходов."""
from dataclasses import dataclass, field
from typing import Callable, Any
from langchain_core.messages import AIMessage


@dataclass
class State:
    """Декларативное описание состояния агента.
    
    Примеры:
        # Простое состояние
        work_state = State(
            name="work",
            tools=["calculator", "ask_human"],
            prompt="Ты помощник для вычислений...",
            description="Основное рабочее состояние"
        )
        
        # С хуками
        def log_entry(state):
            print(f"Вход в состояние: {state}")
            return state
            
        state_with_hooks = State(
            name="special",
            tools=["tool1"],
            prompt="...",
            on_enter=log_entry
        )
    
    Атрибуты:
        name: Уникальное имя состояния (используется в графе)
        tools: Список имен инструментов, доступных в этом состоянии
        prompt: Системный промпт для LLM в этом состоянии
        description: Описание состояния для документации
        on_enter: Функция, вызываемая при входе в состояние (опционально)
        on_exit: Функция, вызываемая при выходе из состояния (опционально)
    """
    name: str
    tools: list[str]
    prompt: str
    description: str = ""
    on_enter: Callable[[dict], dict] | None = None
    on_exit: Callable[[dict], dict] | None = None


@dataclass
class Transition:
    """Описание перехода между состояниями.
    
    Примеры:
        # Простой безусловный переход
        Transition(
            from_state="work",
            to_state="summarize"
        )
        
        # Условный переход (цикл или следующее состояние)
        Transition(
            from_state="work",
            to_state="summarize",
            condition=lambda s: "ГОТОВО" in s['messages'][-1].content
        )
        
        # Роутер (переход в разные состояния в зависимости от условия)
        Transition(
            from_state="router",
            condition=route_by_intent,
            routes={
                "clarify": "clarification_state",
                "compute": "computation_state",
                "summarize": "summary_state"
            }
        )
    
    Атрибуты:
        from_state: Имя состояния, из которого происходит переход
        to_state: Имя целевого состояния (для простых переходов)
        condition: Функция-условие для перехода (принимает state, возвращает bool или str)
        routes: Словарь маршрутов для роутера {результат_условия: целевое_состояние}
        description: Описание перехода для документации
    """
    from_state: str
    to_state: str | None = None
    condition: Callable[[dict], bool | str] | None = None
    routes: dict[str, str] = field(default_factory=dict)
    description: str = ""


class Conditions:
    """Библиотека готовых условий для переходов.
    
    Примеры:
        # Проверка ключевого слова
        Transition(
            from_state="work",
            to_state="summarize",
            condition=Conditions.contains_keyword("ЗАДАЧА_РЕШЕНА")
        )
        
        # Проверка количества сообщений
        Transition(
            from_state="work",
            to_state="timeout",
            condition=Conditions.message_count_exceeds(20)
        )
        
        # Всегда переходить
        Transition(
            from_state="summarize",
            to_state="END",
            condition=Conditions.always_true
        )
    """
    
    @staticmethod
    def contains_keyword(keyword: str, case_sensitive: bool = True) -> Callable:
        """Проверяет наличие ключевого слова в последнем сообщении AI.
        
        Args:
            keyword: Ключевое слово для поиска
            case_sensitive: Учитывать ли регистр
            
        Returns:
            Функция-условие
        """
        def check(state: dict) -> bool:
            messages = state.get('messages', [])
            if not messages:
                return False
            
            last = messages[-1]
            if isinstance(last, AIMessage):
                content = last.content
                search_in = content if case_sensitive else content.lower()
                search_for = keyword if case_sensitive else keyword.lower()
                return search_for in search_in
            return False
        return check
    
    @staticmethod
    def always_true(state: dict) -> bool:
        """Всегда возвращает True (безусловный переход).
        
        Args:
            state: Состояние агента
            
        Returns:
            True
        """
        return True
    
    @staticmethod
    def always_false(state: dict) -> bool:
        """Всегда возвращает False.
        
        Args:
            state: Состояние агента
            
        Returns:
            False
        """
        return False
    
    @staticmethod
    def message_count_exceeds(n: int) -> Callable:
        """Проверяет, превышает ли количество сообщений заданное число.
        
        Args:
            n: Пороговое количество сообщений
            
        Returns:
            Функция-условие
        """
        def check(state: dict) -> bool:
            messages = state.get('messages', [])
            return len(messages) > n
        return check
    
    @staticmethod
    def last_message_is_from_tool(tool_name: str) -> Callable:
        """Проверяет, является ли последнее сообщение от указанного инструмента.
        
        Args:
            tool_name: Имя инструмента
            
        Returns:
            Функция-условие
        """
        def check(state: dict) -> bool:
            from langchain_core.messages import ToolMessage
            messages = state.get('messages', [])
            if not messages:
                return False
            last = messages[-1]
            return isinstance(last, ToolMessage) and last.name == tool_name
        return check
    
    @staticmethod
    def memory_contains(key: str) -> Callable:
        """Проверяет наличие ключа в памяти агента.
        
        Args:
            key: Ключ для проверки
            
        Returns:
            Функция-условие
        """
        def check(state: dict) -> bool:
            memory = state.get('memory', {})
            return key in memory
        return check
    
    @staticmethod
    def combine_and(*conditions: Callable) -> Callable:
        """Логическое И для нескольких условий.
        
        Args:
            *conditions: Условия для объединения
            
        Returns:
            Функция-условие
        """
        def check(state: dict) -> bool:
            return all(cond(state) for cond in conditions)
        return check
    
    @staticmethod
    def combine_or(*conditions: Callable) -> Callable:
        """Логическое ИЛИ для нескольких условий.
        
        Args:
            *conditions: Условия для объединения
            
        Returns:
            Функция-условие
        """
        def check(state: dict) -> bool:
            return any(cond(state) for cond in conditions)
        return check
