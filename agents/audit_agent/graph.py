"""Граф переходов агента аудита проверки."""

from agent_engine import Transition, Conditions, AgentGraphBuilder
from .states import ALL_AUDIT_STATES


def _route_after_self_check(state: dict) -> str:
    memory = state.get("memory", {})
    next_state = memory.get("next_state", "").strip()
    if next_state in {"analize_word", "analize_sql", "analize_py", "write_report"}:
        return next_state
    return "self_check"


transitions = [
    Transition(
        from_state="start_work",
        to_state="analize_word",
        condition=Conditions.contains_keyword("START_WORK_DONE", case_sensitive=False),
        description="Переход после успешного выбора папки и сбора файлов"
    ),
    Transition(
        from_state="analize_word",
        to_state="analize_sql",
        condition=Conditions.contains_keyword("ANALIZE_WORD_DONE", case_sensitive=False),
        description="Переход после анализа docx"
    ),
    Transition(
        from_state="analize_sql",
        to_state="analize_py",
        condition=Conditions.contains_keyword("ANALIZE_SQL_DONE", case_sensitive=False),
        description="Переход после анализа SQL"
    ),
    Transition(
        from_state="analize_py",
        to_state="self_check",
        condition=Conditions.contains_keyword("ANALIZE_PY_DONE", case_sensitive=False),
        description="Переход после анализа Python"
    ),
    Transition(
        from_state="self_check",
        condition=_route_after_self_check,
        routes={
            "analize_word": "analize_word",
            "analize_sql": "analize_sql",
            "analize_py": "analize_py",
            "write_report": "write_report",
            "self_check": "self_check"
        },
        description="Роутер после самопроверки"
    ),
    Transition(
        from_state="write_report",
        to_state="END",
        condition=Conditions.always_true,
        description="Завершение после формирования отчета"
    )
]


def build_audit_agent(llm, tools_dict: dict):
    """Собирает граф агента аудита проверки."""
    builder = AgentGraphBuilder(llm, tools_dict)
    builder.add_states(ALL_AUDIT_STATES)
    builder.add_transitions(transitions)
    builder.set_entry("start_work")
    return builder.build()


def visualize_graph(llm, tools_dict: dict) -> str:
    """Возвращает текстовое описание графа аудита."""
    builder = AgentGraphBuilder(llm, tools_dict)
    builder.add_states(ALL_AUDIT_STATES)
    builder.add_transitions(transitions)
    builder.set_entry("start_work")
    return builder.visualize()
