"""Реестр агентов и универсальные функции сборки."""

from importlib import import_module
from pathlib import Path
import inspect
import pkgutil

from src.agent_engine.base_agent import AgentConfig
from src.tools.tools import get_tools_dict


def _discover_agent_class(agent_module):
    classes = [
        cls for _, cls in inspect.getmembers(agent_module, inspect.isclass)
        if issubclass(cls, AgentConfig)
        and cls is not AgentConfig
        and cls.__module__ == agent_module.__name__
    ]
    if len(classes) != 1:
        raise ValueError(
            f"В модуле {agent_module.__name__} должен быть ровно один класс-наследник AgentConfig"
        )
    return classes[0]


def _get_agent_registry() -> dict[str, type]:
    registry = {}
    package_dir = Path(__file__).resolve().parent
    for module_info in pkgutil.iter_modules([str(package_dir)]):
        if not module_info.ispkg:
            continue
        agent_name = module_info.name
        if agent_name.startswith("_"):
            continue
        module = import_module(f"src.agents.{agent_name}.agent")
        registry[agent_name] = _discover_agent_class(module)
    return registry


def build_agent(agent_name: str, llm):
    """Собирает агент по имени с его tool-набором."""
    registry = _get_agent_registry()
    if agent_name not in registry:
        raise ValueError(f"Неизвестный агент: {agent_name}. Доступные: {', '.join(registry)}")
    agent_cls = registry[agent_name]
    tools_dict = get_tools_dict(agent_name=agent_name)
    return agent_cls(llm, tools_dict)


def visualize_agent(agent_name: str, llm) -> str:
    """Возвращает визуализацию агента по имени."""
    registry = _get_agent_registry()
    if agent_name not in registry:
        raise ValueError(f"Неизвестный агент: {agent_name}. Доступные: {', '.join(registry)}")
    agent_cls = registry[agent_name]
    tools_dict = get_tools_dict(agent_name=agent_name)
    return agent_cls(llm, tools_dict).visualize()


def list_agents() -> list[str]:
    """Возвращает список доступных агентов."""
    return sorted(_get_agent_registry().keys())


__all__ = ["build_agent", "visualize_agent", "list_agents"]
