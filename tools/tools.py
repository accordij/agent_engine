"""Инструменты для агента."""
from langchain.tools import tool


@tool
def calculator(expression: str) -> str:
    """Вычисляет математическое выражение, например: '2 + 3 * 4'"""
    try:
        allowed_names = {"__builtins__": {}}
        result = eval(expression, allowed_names, {})
        return str(result)
    except Exception as e:
        return f"Ошибка вычисления: {e}"


# Экспорт списка инструментов
tools = [calculator]
