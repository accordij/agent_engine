"""Граф переходов математического агента.

Этот файл описывает логику переходов между состояниями:
- Откуда → Куда
- При каких условиях
- В какой последовательности
"""

from agent_engine import Transition, Conditions, AgentGraphBuilder
from .states import ALL_STATES


# ============================================================
# ПЕРЕХОДЫ МЕЖДУ СОСТОЯНИЯМИ
# ============================================================

transitions = [
    # ──────────────────────────────────────────────────────
    # Переход 1: Work → Summarize (когда задача решена)
    # ──────────────────────────────────────────────────────
    # Агент работает циклически в состоянии Work, пока не 
    # скажет "ЗАДАЧА_РЕШЕНА", после чего переходит в Summarize
    # ──────────────────────────────────────────────────────
    Transition(
        from_state="work",
        to_state="summarize",
        condition=Conditions.contains_keyword("ЗАДАЧА_РЕШЕНА", case_sensitive=False),
        description=(
            "Переход из рабочего состояния в состояние подведения итогов. "
            "Срабатывает, когда агент говорит 'ЗАДАЧА_РЕШЕНА'. "
            "Если условие не выполнено, агент остается в Work и продолжает работу."
        )
    ),
    
    # ──────────────────────────────────────────────────────
    # Переход 2: Summarize → END (завершение)
    # ──────────────────────────────────────────────────────
    # После создания саммари работа агента завершается
    # ──────────────────────────────────────────────────────
    Transition(
        from_state="summarize",
        to_state="END",
        condition=Conditions.always_true,
        description=(
            "Безусловное завершение работы после подведения итогов. "
            "Состояние Summarize всегда переходит в END."
        )
    )
]


# ============================================================
# СБОРКА ГРАФА
# ============================================================

def build_my_agent(llm, tools_dict: dict):
    """Собирает граф математического агента.
    
    Args:
        llm: Языковая модель (ChatOpenAI, GigaChat и т.д.)
        tools_dict: Словарь инструментов {имя: объект_инструмента}
        
    Returns:
        Скомпилированный граф LangGraph
        
    Пример:
        from tools.tools import tools
        from connections.clients import get_llm_client
        
        # Подготовка
        llm = get_llm_client("lmstudio", config)
        tools_dict = {tool.name: tool for tool in tools}
        
        # Сборка агента
        agent = build_my_agent(llm, tools_dict)
        
        # Запуск
        result = agent.invoke({
            'messages': ["Вычисли 2+2"],
            'memory': {}
        })
    """
    # Создаем сборщик
    builder = AgentGraphBuilder(llm, tools_dict)
    
    # Добавляем состояния
    builder.add_states(ALL_STATES)
    
    # Добавляем переходы
    builder.add_transitions(transitions)
    
    # Устанавливаем точку входа (с чего начинается граф)
    builder.set_entry("work")
    
    # Собираем и возвращаем граф
    return builder.build()


def visualize_graph(llm, tools_dict: dict) -> str:
    """Возвращает текстовое описание графа агента.
    
    Args:
        llm: Языковая модель
        tools_dict: Словарь инструментов
        
    Returns:
        Строка с описанием состояний и переходов
    """
    builder = AgentGraphBuilder(llm, tools_dict)
    builder.add_states(ALL_STATES)
    builder.add_transitions(transitions)
    builder.set_entry("work")
    
    return builder.visualize()
