# Agent Engine: Quick Start

Короткий гайд по созданию агентов на этом движке.
Цель: за 10-15 минут собрать своего агента с графом состояний, тулзами и памятью.

## 1) Создание агента (`agent.py`)

Базовый паттерн:
- агент наследуется от `AgentConfig`;
- задаются `entry_point` и `states`;
- каждое состояние описывается `State(name, tools, prompt, transitions, memory_injections=...)`;
- переходы делает сам LLM через системный tool `transition` (добавляется движком автоматически).

```python
from src.agent_engine import AgentConfig, State


class MyAgent(AgentConfig):
    entry_point = "work"

    states = [
        State(
            name="work",
            tools=["memory", "think", "calculator"],
            prompt=(
                "Реши задачу пользователя. "
                "Когда закончишь, вызови transition в END."
            ),
            transitions=["END"],
            memory_injections=[
                ("task_context", "Контекст задачи: ", "Контекст пока пуст."),
            ],
            description="Основное рабочее состояние",
        )
    ]
```

Где смотреть реальные примеры:
- `src/agents/test_agent/agent.py` (минимальный агент);
- `src/agents/router_agent/agent.py` (ветвление/роутинг);
- `src/agents/supervisor_agent/agent.py` (композиция агентов).

## 1.1) Мультиагенты (`sub_agents` + `call_agent`)

Для композиции агентов используйте декларативный список `sub_agents` в `agent.py`.
Движок автоматически валидирует имена, собирает и регистрирует этих агентов при сборке графа.

```python
from src.agent_engine import AgentConfig, State


class SupervisorAgent(AgentConfig):
    entry_point = "delegate"
    sub_agents = ["test_agent", "router_agent"]  # авто-регистрация движком

    states = [
        State(
            name="delegate",
            tools=["call_agent", "memory", "think"],
            prompt="Координируй под-агентов и в конце вызови transition.",
            transitions=["aggregate"],
        ),
        State(
            name="aggregate",
            tools=["memory", "summarize", "think"],
            prompt="Собери итог и ОБЯЗАТЕЛЬНО вызови transition(next_state='END').",
            transitions=["END"],
        ),
    ]
```

Важно:
- ручной `register_agent(...)` для `sub_agents` больше не нужен;
- `sub_agents` должен быть `list`/`tuple` непустых строк;
- нельзя добавлять самого себя в `sub_agents`;
- если в конце шага не вызвать `transition(...)`, агент останется в том же состоянии (`stay`/re-entry), что выглядит как "зависание".

## 2) Инструменты: shared vs agent-specific

### Shared инструменты
Базовые тулзы и автодискавери находятся в `src/tools/tools.py`.

- встроенные core-тулзы: `calculator`, `ask_human`, `memory`, `memory_append`, `memory_read`, `summarize`, `think`;
- shared-модули из `src/tools/*.py` подключаются автоматически, если в модуле есть список `TOOLS = [...]`.

### Agent-specific инструменты
Для конкретного агента создайте `src/agents/<agent_name>/tools.py` и экспортируйте:

```python
TOOLS = [
    my_tool_1,
    my_tool_2,
]
```

Подключение происходит через:

```python
tools_dict = get_tools_dict(agent_name="my_agent")
```

Тогда агент получает:
- core shared тулзы;
- shared тулзы из `src/tools/*`;
- только свои `src/agents/my_agent/tools.py`;
- мультиагентные тулзы (например `call_agent`).

Важно: имена тулов должны быть уникальными, иначе будет ошибка валидации.

## 3) Конфиг (`config.yaml`)

Ключевые поля:
- `active_backend`: какой LLM-бэкенд использовать (`gigachat` или `lmstudio`);
- `backends.gigachat` / `backends.lmstudio`: модель, температура, timeout, URL/env;
- `logging.level`: `off | simple | detailed`;
- `logging.renderer`: `off | auto | rich | ansi` (для notebook рекомендуется `auto`);
- `logging.raw_io`: логировать сырой request/response;
- `logging.aggregated`: при `level=detailed` — компактный вывод: `REQ N | msgs=<count> in=<tokens> out=<tokens>` и только дельта контекста (новые сообщения с момента первого отличия от предыдущего запроса). Без повторов `[SYS][USER][tool1]...[answer]` в каждом вызове;
- `logging.filters`: включение/отключение `system/human/tools/assistant/state/memory` отдельно для `global` и `jupyter`;
- `agent.recursion_limit`: лимит шагов графа (используйте при вызове агента через `config`).

Пример альтернативной палитры для light theme (если дефолтная тёмная палитра выглядит бледно):

```yaml
logging:
  colors:
    system: "blue"
    human: "cyan4"
    assistant: "green4"
    reasoning: "spring_green4"
    tool: "gold3"
    tool_name: "bold goldenrod"
    warning: "dark_orange3"
    error: "bold red3"
    state: "bold dark_magenta"
    tokens: "deep_sky_blue4"
    memory: "medium_violet_red"
    run: "bold black"
    info: "grey23"
```

LLM-клиент создается через `src/connections/clients.py`:

```python
llm = get_llm_client(config["active_backend"], config)
```

## 4) Память и контекст между состояниями

### Роль памяти
Память в этом движке — **общее key-value хранилище сессии** (`_memory_store` в `src/tools/tools.py`). Она работает как **клей между состояниями одного агента**: состояние A сохраняет факты под ключами, состояние B читает их тем же ключом без повторного вывода всей истории в промпт.

В **мультиагентных** сценариях (супервизор, `call_agent` и т.д.) агенты в рамках одного процесса и одной сессии обычно делят **ту же** память: один агент записал — другой может прочитать, если у него в графе есть тул `memory` и согласованы имена ключей.

Отдельно есть **журнал** `_memory_log` и тулы `memory_append` / `memory_read` — это не основное хранилище «фактов для графа», а скорее отладочная/текстовая лента.

### Тул `memory`: save / get / list и пакетные вызовы
Чтобы **сократить число вызовов LLM** (каждый вызов тула — отдельный раунд диалога), в `memory` заложены **пакетные** операции:

- **Один save нескольких пар ключ–значение** — передайте `keys` и `values` одинаковой длины (и `action="save"`).
- **Один get нескольких ключей** — `action="get"` и список `keys`; ответ приходит в поле `entries` (словарь ключ → значение или `null`, если ключа нет).
- **Список ключей** — `action="list"`.

Пример пакетного сохранения и чтения (из кода или тестов, не обязательно через LLM):

```python
memory.invoke({
    "action": "save",
    "keys": ["case_id", "case_folder", "status"],
    "values": ["42", "/data/case_42", "ready"],
})
got = memory.invoke({"action": "get", "keys": ["case_id", "status"]})
# got["entries"]["case_id"], got["entries"]["status"]
```

Одиночные `key` / `value` по-прежнему поддерживаются для совместимости.

### Инъекция в промпт при входе в состояние (`memory_injections`)
Чтобы **вообще не тратить раунд LLM** на вызов `memory(get)` при входе в новое состояние, можно **заранее** описать, какие ключи из `_memory_store` подмешать в историю сообщений как отдельные human-сообщения с префиксом «Контекст из памяти:».

В `State(...)` задайте `memory_injections` — поддерживаются форматы из `src/agent_engine/state.py`, например:

- только ключ: `"task_context"` — подставится значение с дефолтным префиксом;
- кортеж `(ключ, текст_если_есть, текст_если_нет)` — как в примере в §1;
- `MemoryInjection(key=..., if_exists=..., if_missing=...)`.

Движок сам читает `_memory_store` и добавляет сообщения при входе в состояние (см. `graph_builder._build_memory_injection_messages`). Ключи в память по-прежнему можно заполнять **вручную** из кода (`memory.invoke(...)`), из других тулов или по инструкции агенту в промпте — инъекция только **доставляет** уже сохранённое в контекст шага.

### Как ещё переносится контекст между состояниями
- **`memory_injections`** — автоматическая подстановка выбранных ключей при входе в state.
- **`transition(..., summary=...)`** — текст резюме попадает в следующий шаг графа; в промптах движка также напоминается список ключей в памяти и вызов `memory(action='get', ...)` при необходимости.

### Практика
- перед новым сценарием вызывайте `reset_memory()`;
- договоритесь о **стабильных именах ключей** между состояниями и агентами (`request_type`, `case_id`, `final_report`, …);
- для экономии раундов: **пакетный** `memory(save/get)` и **`memory_injections`** вместо серии отдельных вызовов `memory` моделью;
- избегайте бесконечных `stay` без новых данных в памяти или в сообщениях.

## 5) Запуск Streamlit UI

```bash
streamlit run src/ui/streamlit_ui.py
```

Откроется браузер с интерфейсом. В сайдбаре выбери агента и введи стартовое сообщение (можно оставить пустым — агент стартует без контекста).

**Доступные агенты и примеры задач:**

| Агент | Граф | Пример стартового сообщения |
|---|---|---|
| `test_agent` | `[work] → END` | `Вычисли 2^10 + 144 / 12` |
| `my_agent` | `[work] → [summarize] → END` | `Посчитай среднее арифметическое чисел 17, 34, 52, 89` или `Нарисуй график продаж по кварталам: Q1=120, Q2=95, Q3=140, Q4=200` |
| `router_agent` | `[classify] → [math\|text\|error] → END` | `Сколько будет 15% от 3200?` или `Объясни что такое рекурсия` |
| `audit_agent` | зависит от реализации | произвольная задача |
| `supervisor_agent` | `[delegate] → [aggregate] → END` | использует `sub_agents` (авто-регистрация движком) |

**Настройка внешнего вида** задаётся в `config.yaml` в секции `streamlit:` — тема (`dark` / `light` / `catppuccin`), размеры блоков, шрифты, частота обновления.

## 6) Вывод в ленту событий: текст и изображения

Из любого инструмента можно напрямую вставить контент в ленту событий Streamlit — без дополнительных вызовов агента.

### `ui_print(text)` — текстовое сообщение

```python
from src.tools.tools import ui_print

ui_print("Промежуточный результат: 42")
```

В терминале выводится всегда; в Streamlit — как отдельный блок `tool`-сообщения в ленте.

### `ui_image(path, caption="")` — изображение

```python
from src.tools.tools import ui_image

ui_image("/abs/path/to/chart.png", caption="Продажи за Q1")
```

Сохраните файл на диск любым способом (matplotlib, PIL, cv2 и т.д.), затем вызовите `ui_image` — картинка сразу появится в ленте событий.

**Правила:**
- путь должен быть доступен процессу Streamlit (абсолютный или относительный от корня проекта);
- если файл не найден — в ленте появится предупреждение вместо картинки;
- `ui_image` ничего не возвращает и не блокирует — вызов безопасен вне UI-режима.

### Пример: инструмент с графиком

Полный рабочий пример — `src/agents/my_agent/tools.py`, инструмент `plot_chart`:

```python
import matplotlib
matplotlib.use("Agg")           # backend устанавливается один раз на уровне модуля
import matplotlib.pyplot as plt

from src.tools.tools import ui_image

@tool
def plot_chart(title: str, labels: str, values: str, chart_type: str = "bar") -> str:
    """Строит график и показывает его пользователю в интерфейсе.

    После успешного построения вызови transition для перехода к следующему состоянию.
    """
    # ... парсинг, построение графика ...
    plt.savefig(str(filepath), dpi=150, bbox_inches="tight")
    plt.close(fig)

    ui_image(str(filepath), caption=title)      # → картинка в ленте Streamlit
    return f"График '{title}' построен и отображён пользователю"
```

**Важно:** `matplotlib.use("Agg")` и `import matplotlib.pyplot as plt` должны быть на уровне модуля, а не внутри функции. При переносе внутрь функции на повторных вызовах matplotlib сбрасывает состояние и сохраняет пустой белый файл.

### Регистрация инструмента в агенте

```python
# src/agents/my_agent/tools.py
TOOLS = [plot_chart]

# src/agents/my_agent/agent.py
State(
    name="work",
    tools=["memory", "think", "plot_chart"],
    prompt="... 7. Вызови transition(next_state='summarize') — обязательный последний шаг",
    transitions=["summarize"],
)
```

Тестовый запрос для `my_agent`: `Нарисуй столбчатый график продаж по кварталам: Q1=120, Q2=95, Q3=140, Q4=200`

## 7) Минимальный запуск (из кода)

```python
import yaml
from src.agent_engine import init_logging
from src.connections.clients import get_llm_client
from src.tools.tools import get_tools_dict, reset_memory
from src.agents.my_agent import MyAgent

with open("config.yaml", "r", encoding="utf-8") as f:
    config = yaml.safe_load(f)

init_logging()
llm = get_llm_client(config["active_backend"], config)
tools_dict = get_tools_dict(agent_name="my_agent")
reset_memory()

agent = MyAgent(llm, tools_dict)
result = agent.invoke(["Реши задачу пользователя"], config={"recursion_limit": config["agent"]["recursion_limit"]})
print(result["messages"][-1].content)
```

## 8) Частые ошибки

- `entry_point` не совпадает с именем состояния в `states`.
- В `transitions` указан state, которого нет в графе.
- Тул из `State.tools` отсутствует в `tools_dict`.
- Дублируются имена тулов в разных модулях (`TOOLS`).
- Память не сброшена (`reset_memory()`), и старый контекст влияет на новый запуск.
- Агент уходит в повторные `stay` без обновления данных в памяти.
