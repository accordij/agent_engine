# Управление историей сообщений в LangChain/LangGraph

> Справочный документ по практикам управления контекстным окном агента, передаче информации между состояниями и встроенным возможностям.

---

## 1. Понимание проблемы

### Симптомы

- Агент получает большое количество сообщений типа `ToolMessage`
- Агент галлюцинирует, уходит в цикл
- Штампует большое количество пустых сообщений
- Ошибка `INVALID_CHAT_HISTORY`: AIMessages с `tool_calls` без соответствующих `ToolMessage`

### Корень проблемы

1. **Переполнение контекста** — слишком много сообщений превышает эффективное окно модели
2. **Шум от ToolMessage** — старые результаты инструментов отвлекают модель
3. **Нарушение валидности истории** — LLM-провайдеры требуют: каждому `tool_call` в AIMessage должен соответствовать `ToolMessage` с результатом

### Золотое правило

**Нельзя удалять ToolMessage, оставляя AIMessage с tool_calls** — это нарушает контракт и приводит к ошибке. Либо пара (AIMessage + ToolMessage) целиком, либо никак.

---

## 2. Мировая практика: четыре подхода

Согласно документации LangGraph:

| Подход | Суть | Плюсы | Минусы |
|--------|------|-------|--------|
| **Trim** | Обрезка по токенам (первые/последние N) | Просто, быстро | Теряется контекст |
| **Summarize** | Суммаризировать старые сообщения, заменить на summary | Сохраняется смысл | Дорого, медленно, risk потери деталей |
| **Delete** | Удалять сообщения по ID или всё сразу | Полный контроль | Нужно самим валидировать историю |
| **Custom filter** | Своя логика фильтрации перед вызовом LLM | Максимальная гибкость | Больше кода |

**Рекомендация LangGraph:** логику управления историей выполнять **в узле перед вызовом модели**, а не в общем хранилище state.

---

## 3. Что оставлять и что убирать

### Принцип парности

Любое изменение должно сохранять **целостность tool-звеньев**:
- AIMessage с `tool_calls` → обязательно идут следом ToolMessage(s) с результатами
- Либо удаляете пару целиком, либо оставляете

### Вариант A: Сжатие старых tool-звеньев

Сжать старые пары (AIMessage + ToolMessage) в короткое HumanMessage-резюме:
> "Результат: агент посчитал 2^10 = 1024, спросил пользователя о радиусе"

### Вариант B: Trim с правильными границами (рекомендуется)

Использовать `trim_messages` — он умеет обрезать по токенам, **сохраняя валидность** (не режет посередине tool-звена):

```python
from langchain_core.messages.utils import trim_messages, count_tokens_approximately

messages = trim_messages(
    state["messages"],
    strategy="last",
    token_counter=count_tokens_approximately,
    max_tokens=4000,
    start_on="human",
    end_on=("human", "tool"),
)
```

### Вариант C: Разный контекст для разных состояний

- **Work** — полная история (или trim)
- **Summarize** — передавать только сводку/результат работы, а не сырую историю с десятками ToolMessage

---

## 4. Встроенные функции LangChain/LangGraph

### `trim_messages` (langchain_core)

**Импорт:**
```python
from langchain_core.messages.utils import trim_messages, count_tokens_approximately
```

**Основные параметры:**
- `strategy` — `"last"` (оставить последние N токенов) или `"first"`
- `max_tokens` — лимит токенов
- `token_counter` — функция подсчёта (например `count_tokens_approximately`)
- `start_on` — тип сообщения, с которого начинать отсчёт (`"human"`)
- `end_on` — границы обрезки: `("human", "tool")` — не резать посередине tool-звена

**Пример:**
```python
trimmed = trim_messages(
    messages,
    strategy="last",
    token_counter=count_tokens_approximately,
    max_tokens=4000,
    start_on="human",
    end_on=("human", "tool"),
)
```

### `SummarizationNode` (langmem)

Автоматическая суммаризация при превышении порога токенов.

```python
from langmem.short_term import SummarizationNode

summarization_node = SummarizationNode(
    token_counter=count_tokens_approximately,
    model=summarization_model,
    max_tokens=256,
    max_tokens_before_summary=256,
    max_summary_tokens=128,
)
```

⚠️ Известны баги: может оставлять AIMessage с tool_calls без ToolMessage — осторожно.

### `RemoveMessage` (удаление по ID)

```python
from langchain.messages import RemoveMessage
from langgraph.graph.message import REMOVE_ALL_MESSAGES

# Удалить первые 2 сообщения
return {"messages": [RemoveMessage(id=m.id) for m in state["messages"][:2]]}

# Удалить все
return {"messages": [RemoveMessage(id=REMOVE_ALL_MESSAGES)]}
```

При удалении нужно сохранять валидность: пары AIMessage+ToolMessage удалять целиком.

---

## 5. Рекомендации для multi-state агента

1. **Исправить текущий баг:** не удалять только ToolMessage — либо оставлять пары (AIMessage + ToolMessage), либо сжимать/удалять их целиком.

2. **Для summarize:** не передавать сырую историю с кучей ToolMessage. Лучше:
   - Собрать итог работы work-агента в один текст
   - Передать в summarize только этот итог + новый промпт

3. **Для work:** использовать `trim_messages` перед вызовом LLM с `end_on=("human", "tool")`.

4. **Архитектура:** добавить узел/логику между work → summarize, который подготавливает «чистый» контекст для summarize (без сырых ToolMessage).

---

## 6. Полезные ссылки

- [LangGraph Memory (docs)](https://langchain-ai.github.io/langgraph/how-tos/memory/)
- [LangChain trim_messages](https://python.langchain.com/docs/how_to/trim_messages)
- [Troubleshooting INVALID_CHAT_HISTORY](https://python.langchain.com/docs/troubleshooting/errors/INVALID_CHAT_HISTORY)

---

*Документ создан для проекта 23_Agent_CDO3*
