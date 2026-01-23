"""Реестр агентов и универсальные функции сборки."""

from typing import Callable


def _get_agent_registry() -> dict[str, tuple[Callable, Callable]]:
    from agents.my_agent.graph import build_my_agent, visualize_graph as visualize_my
    from agents.audit_agent.graph import build_audit_agent, visualize_graph as visualize_audit

    return {
        "my_agent": (build_my_agent, visualize_my),
        "audit_agent": (build_audit_agent, visualize_audit),
    }


def build_agent(agent_name: str, llm, tools_dict: dict):
    """Собирает агент по имени."""
    registry = _get_agent_registry()
    if agent_name not in registry:
        raise ValueError(f"Неизвестный агент: {agent_name}. Доступные: {', '.join(registry)}")
    builder, _ = registry[agent_name]
    return builder(llm, tools_dict)


def visualize_agent(agent_name: str, llm, tools_dict: dict) -> str:
    """Возвращает визуализацию агента по имени."""
    registry = _get_agent_registry()
    if agent_name not in registry:
        raise ValueError(f"Неизвестный агент: {agent_name}. Доступные: {', '.join(registry)}")
    _, visualizer = registry[agent_name]
    return visualizer(llm, tools_dict)


def list_agents() -> list[str]:
    """Возвращает список доступных агентов."""
    return sorted(_get_agent_registry().keys())


__all__ = ["build_agent", "visualize_agent", "list_agents"]
