# Простой агент с переключаемым бэкендом

Проект простого ReAct-агента с поддержкой двух типов подключения: GigaChat и LM Studio.

## Структура проекта

```
23_Agent_CDO3/
├── main_agent.ipynb          # Jupyter ноутбук для запуска и тестирования агента
├── config.yaml                # Файл конфигурации (выбор бэкенда и настройки)
├── requirements.txt           # Зависимости проекта
├── README.md                  # Документация проекта
├── tools/
│   └── tools.py              # Инструменты агента (калькулятор)
├── connections/
│   └── clients.py            # Модуль подключений к LLM (GigaChat, LM Studio)
└── examples/
    └── gigachat api.ipynb    # Примеры работы с GigaChat API
```

## Возможности

- **Два типа подключения:**
  - **GigaChat** — для работы в корпоративной среде без интернета
  - **LM Studio** — для локальной работы дома через OpenAI-совместимый API
- **Инструменты:**
  - Калькулятор для математических вычислений

#### Пример 1: Простое вычисление
```python
query = "Сколько будет 52 умножить на 48?"
messages = agent_executor.invoke({'messages': [query]})['messages']
print(messages[-1].content)
```

#### Пример 2: Сложное выражение
```python
query = "Вычисли (25 + 17) * 3 - 10"
messages = agent_executor.invoke({'messages': [query]})['messages']
print(messages[-1].content)
```

#### Пример 3: Прямой вызов инструмента
```python
from tools.tools import calculator
result = calculator.invoke("2 ** 10")
print(result)  # 1024
```

## Добавление новых инструментов

Чтобы добавить новый инструмент, отредактируйте `tools/tools.py`:

```python
from langchain.tools import tool

@tool
def my_new_tool(input: str) -> str:
    """Описание инструмента."""
    # Ваша логика
    return result

# Добавьте в список инструментов
tools = [calculator, my_new_tool]
```

## Примечания

- GigaChat использует rate limiting с задержкой 6 секунд между запросами
- LM Studio не требует API ключ для локальной работы
- Все подключения инкапсулированы в модуле `connections/clients.py`