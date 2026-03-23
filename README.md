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

### Как устроена память
- глобальное key-value хранилище: `_memory_store`;
- append-only журнал: `_memory_log`;
- сброс перед новой сессией: `reset_memory()`.

Все это реализовано в `src/tools/tools.py`.

### Как память попадает в следующий state
- через `memory_injections` в `State(...)`;
- через `summary`, которое передается при `transition(next_state=..., summary=...)`.

Практика:
- перед запуском нового сценария вызывайте `reset_memory()`;
- сохраняйте значения под стабильными ключами (`request_type`, `result`, `final_report`);
- избегайте `stay` без новой информации, чтобы не зациклиться.

## 5) Минимальный запуск

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

## 6) Частые ошибки

- `entry_point` не совпадает с именем состояния в `states`.
- В `transitions` указан state, которого нет в графе.
- Тул из `State.tools` отсутствует в `tools_dict`.
- Дублируются имена тулов в разных модулях (`TOOLS`).
- Память не сброшена (`reset_memory()`), и старый контекст влияет на новый запуск.
- Агент уходит в повторные `stay` без обновления данных в памяти.
