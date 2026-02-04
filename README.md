# 🤖 Фреймворк для создания Multi-Agent систем

Проект содержит **фреймворк для построения агентов** с декларативным описанием состояний и переходов. Поддерживает как простых агентов, так и сложные мультиагентные системы.

## ✨ Ключевые особенности

- 🎯 **Один файл - один агент**: Все состояния и переходы в одном месте
- 🔄 **Легкое копирование**: Скопировал папку → переименовал → работает
- 🤝 **Мультиагентность**: Агенты могут вызывать друг друга
- 📊 **Декларативность**: Описываешь ЧТО делать, а не КАК
- 🔍 **Логирование**: Встроенное отслеживание всех шагов

## 🏗️ Архитектура проекта

```
23_Agent_CDO3/
├── agent_engine/              # 🔧 ДВИЖОК (переиспользуемый)
│   ├── __init__.py
│   ├── state.py              # State, Transition, Conditions
│   ├── graph_builder.py      # AgentGraphBuilder
│   ├── base_agent.py         # AgentConfig (базовый класс)
│   └── debug.py              # Логирование
│
├── agents/                    # 🧩 НАБОР АГЕНТОВ
│   ├── test_agent/           # Простейший (1 состояние)
│   │   ├── agent.py         # ВСЁ В ОДНОМ ФАЙЛЕ!
│   │   └── __init__.py
│   ├── my_agent/            # С переходами (2 состояния)
│   │   ├── agent.py
│   │   └── __init__.py
│   ├── router_agent/        # С роутингом (ветвление)
│   │   ├── agent.py
│   │   └── __init__.py
│   ├── supervisor_agent/    # Мультиагент (координатор)
│   │   ├── agent.py
│   │   └── __init__.py
│   └── audit_agent/         # Полный workflow (6 состояний)
│       ├── agent.py
│       └── __init__.py
│
├── tools/
│   └── tools.py              # Инструменты + реестр агентов
│
├── connections/
│   └── clients.py            # LLM клиенты (GigaChat, LM Studio)
│
├── main_agent.ipynb          # 🧪 ТЕСТЫ (6 секций)
├── config.yaml               # Конфигурация LLM
└── data/                     # Данные для audit_agent
```

## 🚀 Быстрый старт

### 1. Установка

```bash
pip install -r requirements.txt
```

### 2. Настройка LLM

Отредактируйте `config.yaml`:

```yaml
active_backend: lmstudio  # или gigachat

backends:
  lmstudio:
    base_url: http://localhost:1234/v1
  gigachat:
    env_vars:
      access_token: JPY_API_TOKEN
```

### 3. Запуск тестов

Откройте `main_agent.ipynb` и запустите все ячейки. Ноутбук содержит 6 секций тестов:

1. ✅ **Тест базовых функций** - проверка инструментов
2. ✅ **Test Agent** - простейший агент (1 состояние)
3. ✅ **My Agent** - переходы между состояниями
4. ✅ **Router Agent** - условный роутинг
5. ✅ **Multi-Agent** - композиция агентов
6. ✅ **Audit Agent** - полный workflow

## 📖 Создание своего агента

### Способ 1: Копирование существующего

```bash
# 1. Скопируйте папку агента
cp -r agents/test_agent agents/my_new_agent

# 2. Отредактируйте agents/my_new_agent/agent.py
# 3. Готово!
```

### Способ 2: Создание с нуля

```python
# agents/my_new_agent/agent.py

from agent_engine import AgentConfig, State, Transition, Conditions


class MyNewAgent(AgentConfig):
    """Описание вашего агента."""
    
    entry_point = "work"  # Начальное состояние
    
    states = [
        State(
            name="work",
            tools=["calculator", "memory", "think"],
            prompt="""Ты помощник для...""",
            description="Основное состояние"
        )
    ]
    
    transitions = [
        Transition(
            from_state="work",
            to_state="END",
            condition=Conditions.contains_keyword("ГОТОВО"),
            description="Завершение работы"
        )
    ]
```

```python
# agents/my_new_agent/__init__.py

from .agent import MyNewAgent

__all__ = ["MyNewAgent"]
```

### Использование агента

```python
from agents.my_new_agent import MyNewAgent
from tools.tools import get_tools_dict, tools, reset_memory
from connections.clients import get_llm_client

# Настройка
llm = get_llm_client("lmstudio", config)
tools_dict = get_tools_dict(tools)
reset_memory()

# Создание агента
agent = MyNewAgent(llm, tools_dict)

# Визуализация графа
print(agent.visualize())

# Запуск
result = agent.invoke(["Посчитай 2+2"])
print(result['messages'][-1].content)
```

## 🎯 Примеры агентов

### 1. Простейший агент (1 состояние)

```python
class TestAgent(AgentConfig):
    entry_point = "work"
    
    states = [
        State(
            name="work",
            tools=["calculator", "memory"],
            prompt="Выполни вычисления и скажи ГОТОВО"
        )
    ]
    
    transitions = [
        Transition(from_state="work", to_state="END",
                  condition=Conditions.contains_keyword("ГОТОВО"))
    ]
```

### 2. Агент с циклом

```python
class MyAgent(AgentConfig):
    entry_point = "work"
    
    states = [
        State(name="work", tools=["calculator", "memory"],
              prompt="Работай пока не решишь задачу"),
        State(name="summarize", tools=["summarize"],
              prompt="Подведи итоги")
    ]
    
    transitions = [
        # work → work (цикл, пока не скажет "ЗАДАЧА_РЕШЕНА")
        Transition(from_state="work", to_state="summarize",
                  condition=Conditions.contains_keyword("ЗАДАЧА_РЕШЕНА")),
        Transition(from_state="summarize", to_state="END",
                  condition=Conditions.always_true)
    ]
```

### 3. Агент с роутингом

```python
def route_by_type(state: dict) -> str:
    memory = state.get("memory", {})
    request_type = memory.get("request_type", "")
    return "math" if request_type == "math" else "text"


class RouterAgent(AgentConfig):
    entry_point = "classify"
    
    states = [
        State(name="classify", tools=["think", "memory"],
              prompt="Определи тип запроса и сохрани в memory"),
        State(name="math", tools=["calculator"],
              prompt="Реши математическую задачу"),
        State(name="text", tools=["think"],
              prompt="Ответь на текстовый запрос")
    ]
    
    transitions = [
        # Роутер: classify → [math | text]
        Transition(
            from_state="classify",
            condition=route_by_type,
            routes={"math": "math", "text": "text"}
        ),
        Transition(from_state="math", to_state="END",
                  condition=Conditions.always_true),
        Transition(from_state="text", to_state="END",
                  condition=Conditions.always_true)
    ]
```

### 4. Мультиагентная система

```python
# Создаем агентов
test_agent = TestAgent(llm, tools_dict)
router_agent = RouterAgent(llm, tools_dict)

# Регистрируем их
from tools.tools import register_agent
register_agent("test_agent", test_agent)
register_agent("router_agent", router_agent)

# Создаем супервизора с инструментом call_agent
from tools.tools import multiagent_tools
supervisor_tools_dict = get_tools_dict(tools + multiagent_tools)

supervisor = SupervisorAgent(llm, supervisor_tools_dict)

# Супервизор может вызывать других агентов!
result = supervisor.invoke([
    "Вызови test_agent для вычислений и router_agent для приветствия"
])
```

## 🛠️ Инструменты

### Базовые инструменты

- **calculator** - вычисление математических выражений
- **ask_human** - задать вопрос пользователю
- **memory** - сохранение/чтение данных (action: save/get/list)
- **memory_append** - добавить строку в журнал
- **memory_read** - прочитать журнал
- **think** - внутренние размышления агента
- **summarize** - создание саммари работы

### Инструменты для файлов (audit_agent)

- **list_data_folders** - список папок
- **find_case_folder** - поиск папки по номеру
- **list_case_files** - список файлов в папке
- **read_docx_structure** - анализ docx
- **read_sql_file** - чтение SQL
- **read_py_file** - чтение Python

### Мультиагентные инструменты

- **call_agent** - вызов зарегистрированного агента

## 📚 Библиотека условий (Conditions)

```python
from agent_engine import Conditions

# Ключевое слово в последнем сообщении
Conditions.contains_keyword("ГОТОВО", case_sensitive=False)

# Количество сообщений
Conditions.message_count_exceeds(20)

# Безусловный переход
Conditions.always_true

# Проверка памяти
Conditions.memory_contains("result")

# Комбинирование
Conditions.combine_and(
    Conditions.contains_keyword("ГОТОВО"),
    Conditions.memory_contains("result")
)
```

## 🎨 Продвинутые возможности

### Хуки состояний

```python
def log_entry(state):
    print(f"🔔 Вход в состояние")
    return state

def validate_exit(state):
    if 'result' not in state.get('memory', {}):
        print("⚠️ Результат не сохранен!")
    return state

State(
    name="work",
    tools=["calculator"],
    prompt="...",
    on_enter=log_entry,
    on_exit=validate_exit
)
```

### Управление памятью

**Вариант 1: Глобальная память** (текущая реализация)
```python
# Все агенты используют общую память через инструменты
reset_memory()  # Очистка перед новой сессией
```

**Вариант 2: Изолированная память** (при необходимости)
```python
agent1 = MyAgent(llm, tools_dict, agent_id="agent_1")
agent2 = MyAgent(llm, tools_dict, agent_id="agent_2")
# Каждый агент имеет свой agent_id для изоляции
```

### Логирование

```python
from agent_engine.debug import enable_logging, disable_logging

# Включить логирование
enable_logging()

# Теперь видны все шаги:
# [STATE] work -> summarize
# [TOOL] calculator params={"expression": "2+2"}
# [TOOL] memory params={"action": "save", "key": "result", "value": "4"}

# Отключить логирование
disable_logging()
```

## 🔄 Workflow агента

```
1. Пользователь → agent.invoke(["Запрос"])
2. Граф начинается с entry_point
3. Для каждого состояния:
   - Выполняется on_enter hook (если есть)
   - Агент получает промпт и инструменты
   - LLM генерирует ответ и вызывает инструменты
   - Выполняется on_exit hook (если есть)
4. Проверяются условия переходов (Transition.condition)
5. Переход в следующее состояние или END
6. Возврат финального состояния с messages и memory
```

## 🧪 Тестирование

Все тесты в `main_agent.ipynb`:

```python
# Секция 1: Базовые функции
calculator.invoke("2 ** 10")  # → 1024

# Секция 2: Test Agent
test_agent = TestAgent(llm, tools_dict)
result = test_agent.invoke(["Посчитай 15 * 23"])

# Секция 3: My Agent
my_agent = MyAgent(llm, tools_dict)
result = my_agent.invoke(["Посчитай 2^10"])

# Секция 4: Router Agent
router_agent = RouterAgent(llm, tools_dict)
result = router_agent.invoke(["Посчитай 5+5"])  # → math path

# Секция 5: Multi-Agent
register_agent("test_agent", test_agent)
supervisor = SupervisorAgent(llm, supervisor_tools_dict)
result = supervisor.invoke(["Делегируй вычисление test_agent"])

# Секция 6: Audit Agent
audit_agent = AuditAgent(llm, tools_dict)
result = audit_agent.invoke(["Проверь папку 99-41116"])
```

## 🎓 Лучшие практики

1. **Один файл на агента**: Все в `agent.py` - легко читать и поддерживать
2. **Явные ключевые слова**: Используйте четкие маркеры для переходов ("ГОТОВО", "ЗАДАЧА_РЕШЕНА")
3. **Описания**: Добавляйте description к State и Transition для документации
4. **Изоляция памяти**: Очищайте память через `reset_memory()` перед новой сессией
5. **Логирование**: Включайте для отладки, отключайте для production
6. **Роутеры в функциях**: Выносите сложные роутеры в отдельные функции

## 📝 Примечания

- **GigaChat**: rate limiting 6 секунд между запросами
- **LM Studio**: не требует API ключ для локальной работы
- **История сообщений**: изолирована для каждого агента
- **Память**: глобальная через инструменты memory (можно изолировать через agent_id)
- **Recursion limit**: настраивается в config.yaml (по умолчанию 300)

## 🤝 Поддержка

Для вопросов и предложений создавайте issues в репозитории.

## 📄 Лицензия

MIT License
