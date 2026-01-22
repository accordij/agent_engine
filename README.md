# 🤖 Движок для создания агентов с графом состояний

Проект содержит **движок для построения агентов** с декларативным описанием состояний и переходов. Включает готовый математический агент как пример использования.

## 🏗️ Архитектура проекта

```
23_Agent_CDO3/
├── agent_engine/              # 🔧 ДВИЖОК (переиспользуемый)
│   ├── __init__.py
│   ├── state.py              # Классы State, Transition, Conditions
│   └── graph_builder.py      # AgentGraphBuilder - сборщик графа
│
├── my_agent/                  # 📝 КОНКРЕТНЫЙ АГЕНТ (ваш агент)
│   ├── __init__.py
│   ├── states.py             # Описание состояний агента
│   └── graph.py              # Граф переходов агента
│
├── tools/
│   └── tools.py              # Инструменты (calculator, ask_human, memory и др.)
│
├── connections/
│   └── clients.py            # LLM клиенты (GigaChat, LM Studio)
│
├── main_agent.ipynb          # Примеры и тесты
├── config.yaml               # Конфигурация LLM
└── examples/
    └── gigachat api.ipynb    # Примеры работы с GigaChat API
```

### Разделение ответственности

- **`agent_engine/`** — универсальный движок (не трогаем при создании новых агентов)
- **`my_agent/`** — конкретный агент (создаем/редактируем для новых агентов)
- **`tools/`** — библиотека инструментов (расширяем по мере необходимости)

## ✨ Возможности

### 🔌 Два типа подключения к LLM
- **GigaChat** — для работы в корпоративной среде без интернета
- **LM Studio** — для локальной работы дома через OpenAI-совместимый API

### 🎯 Движок для построения агентов
- Декларативное описание состояний (класс `State`)
- Декларативное описание переходов (класс `Transition`)
- Библиотека готовых условий (`Conditions`)
- Fluent API для сборки графа (`AgentGraphBuilder`)
- Хуки `on_enter`/`on_exit` для кастомной логики

### 🛠️ Готовые инструменты
- **calculator** — вычисление математических выражений
- **ask_human** — задавать уточняющие вопросы пользователю
- **memory** — сохранение и чтение заметок в памяти агента
- **think** — внутренние размышления агента
- **summarize** — создание саммари выполненной работы

## 🚀 Workflow создания нового агента

### Шаг 1: Создайте инструменты (если нужны новые)

```python
# tools/tools.py

@tool
def my_new_tool(input: str) -> str:
    """Описание инструмента."""
    # Ваша логика
    return result

tools = [calculator, ask_human, memory, think, summarize, my_new_tool]
```

### Шаг 2: Опишите состояния

```python
# my_agent/states.py

from agent_engine import State

work_state = State(
    name="work",
    tools=["calculator", "ask_human", "my_new_tool"],
    prompt="Ты помощник для...",
    description="Основное рабочее состояние"
)

summarize_state = State(
    name="summarize",
    tools=["summarize", "memory"],
    prompt="Ты подводишь итоги...",
    description="Финальное состояние"
)

ALL_STATES = [work_state, summarize_state]
```

### Шаг 3: Опишите граф переходов

```python
# my_agent/graph.py

from agent_engine import Transition, Conditions, AgentGraphBuilder
from .states import ALL_STATES

transitions = [
    Transition(
        from_state="work",
        to_state="summarize",
        condition=Conditions.contains_keyword("ГОТОВО"),
        description="Переход к саммари"
    ),
    Transition(
        from_state="summarize",
        to_state="END",
        condition=Conditions.always_true
    )
]

def build_my_agent(llm, tools_dict):
    builder = AgentGraphBuilder(llm, tools_dict)
    return (builder
        .add_states(ALL_STATES)
        .add_transitions(transitions)
        .set_entry("work")
        .build())
```

### Шаг 4: Используйте агента

```python
from my_agent import build_my_agent

# Подготовка
tools_dict = {tool.name: tool for tool in tools}

# Создание
agent = build_my_agent(llm, tools_dict)

# Запуск
result = agent.invoke({'messages': ["Задача"], 'memory': {}})
```

## Примеры использования

### Простой ReAct агент (базовый)

#### Пример 1: Простое вычисление
```python
query = "Сколько будет 52 умножить на 48?"
messages = agent_executor.invoke({'messages': [query]})['messages']
print(messages[-1].content)
```

#### Пример 3: Прямой вызов инструмента
```python
from tools.tools import calculator
result = calculator.invoke("2 ** 10")
print(result)  # 1024
```

### Примеры с готовым математическим агентом

#### Пример 1: Простое вычисление с памятью
```python
from my_agent import build_my_agent
from tools.tools import tools

# Подготовка
tools_dict = {tool.name: tool for tool in tools}
agent = build_my_agent(llm, tools_dict)

# Запрос
result = agent.invoke({
    'messages': ["Вычисли 25 * 4 и сохрани результат в память"],
    'memory': {}
})

print(result['messages'][-1].content)
print(result['memory'])  # Сохраненные данные
```

#### Пример 2: Вычисление с уточнениями
```python
# Агент задаст вопросы через ask_human
result = agent.invoke({
    'messages': ["Вычисли площадь прямоугольника. Если не хватает данных, спроси."],
    'memory': {}
})

# В процессе выполнения агент задаст вопросы:
# 🤔 Вопрос агента: Какая длина прямоугольника?
# 👤 Ваш ответ: 10
# 🤔 Вопрос агента: Какая ширина?
# 👤 Ваш ответ: 5

print(result['messages'][-1].content)  # Площадь = 50
```

#### Пример 3: Переход между состояниями
```python
result = agent.invoke({
    'messages': ["""Реши задачу пошагово:
    1. Используй think чтобы поразмышлять
    2. Вычисли (15 + 25) * 2
    3. Сохрани результат в память
    4. Скажи ЗАДАЧА_РЕШЕНА"""],
    'memory': {}
})

# Агент автоматически пройдет через оба состояния:
# Work State → размышление, вычисление, сохранение → "ЗАДАЧА_РЕШЕНА"
# Summarize State → создание саммари
```

## 📚 Библиотека условий (Conditions)

Движок предоставляет готовые условия для переходов:

```python
from agent_engine import Conditions

# Проверка ключевого слова
Conditions.contains_keyword("ГОТОВО", case_sensitive=False)

# Проверка количества сообщений
Conditions.message_count_exceeds(20)

# Безусловный переход
Conditions.always_true

# Проверка памяти
Conditions.memory_contains("result")

# Проверка последнего инструмента
Conditions.last_message_is_from_tool("calculator")

# Комбинирование условий
Conditions.combine_and(
    Conditions.contains_keyword("ГОТОВО"),
    Conditions.memory_contains("result")
)
```

## 🎨 Продвинутые возможности

### Хуки состояний

Добавьте кастомную логику при входе/выходе из состояния:

```python
def log_entry(state):
    print(f"🔔 Вход в состояние, сообщений: {len(state['messages'])}")
    return state

def validate_exit(state):
    # Проверяем, что результат сохранен
    if 'result' not in state.get('memory', {}):
        print("⚠️ Результат не сохранен!")
    return state

my_state = State(
    name="work",
    tools=["calculator"],
    prompt="...",
    on_enter=log_entry,
    on_exit=validate_exit
)
```

### Роутер (множественные переходы)

Переход в разные состояния в зависимости от результата:

```python
def route_by_intent(state) -> str:
    """Определяет намерение пользователя."""
    last_msg = state['messages'][-1].content.lower()
    
    if "вопрос" in last_msg or "непонятно" in last_msg:
        return "clarification"
    elif "вычисли" in last_msg or "посчитай" in last_msg:
        return "computation"
    else:
        return "general"

Transition(
    from_state="router",
    condition=route_by_intent,
    routes={
        "clarification": "clarification_state",
        "computation": "computation_state",
        "general": "general_state"
    }
)
```

### Визуализация графа

```python
from my_agent.graph import visualize_graph

print(visualize_graph(llm, tools_dict))

# Выведет:
# 📊 Граф агента:
# 
# Состояния:
#   • work (entry)
#     Инструменты: calculator, ask_human, think, memory
#   • summarize
#     Инструменты: summarize, memory
# 
# Переходы:
#   • work → summarize (условный)
#     Переход к саммари когда задача решена
#   • summarize → END
```

## Примечания

- GigaChat использует rate limiting с задержкой 6 секунд между запросами
- LM Studio не требует API ключ для локальной работы
- Все подключения инкапсулированы в модуле `connections/clients.py`