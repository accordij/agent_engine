"""Базовый класс для конфигурации агентов."""
from __future__ import annotations


class NothingToResumeError(Exception):
    """Сессия уже завершилась — нечего продолжать.

    Бросается из resume() когда граф дошёл до END.
    Используй другой checkpoint или создай форк через fork().

    Атрибуты:
        session_id: ID сессии которую пытались возобновить.
    """

    def __init__(self, session_id: str) -> None:
        self.session_id = session_id
        super().__init__(
            f"Сессия '{session_id[:8]}...' уже завершена (END). "
            "Восстанавливать нечего. "
            "Выбери другой checkpoint или создай форк через fork()."
        )

import contextvars
from pathlib import Path
from typing import Any, Dict

from .graph_builder import AgentGraphBuilder
from .state import State
from .logging_utils import create_callbacks, log_run_start, log_run_end

# Хранит session_id текущего активного вызова invoke().
# ContextVar — thread-safe: каждый поток имеет свой контекст,
# поэтому supervisor и sub-агент в одном потоке видят разные значения
# (sub-агент перекрывает значение своим session_id, но не затирает supervisor-а).
_active_session_ctx: contextvars.ContextVar[str | None] = contextvars.ContextVar(
    "_active_session_ctx", default=None
)


def _load_sessions_config() -> dict:
    """Читает секцию sessions из config.yaml. Возвращает {} при ошибке."""
    try:
        import yaml
        config_path = Path(__file__).parent.parent.parent / "config.yaml"
        if not config_path.exists():
            return {}
        with open(config_path, encoding="utf-8") as f:
            cfg = yaml.safe_load(f)
        return cfg.get("sessions", {})
    except Exception:
        return {}


# Синглтон: один SessionManager на db_path — все агенты делят одно соединение.
# Ключ — абсолютный путь к БД чекпоинтов.
_SM_REGISTRY: dict[str, Any] = {}


def _make_session_manager():
    """Вернуть SessionManager из config.yaml (синглтон по db_path).

    Все AgentConfig с одинаковым db_path используют один объект и одно
    SQLite-соединение. Это важно для мультиагентных схем: sub-агенты
    создаются внутри run-а supervisor-а и не должны порождать лишние
    соединения и лишние вызовы prune_on_load().
    """
    cfg = _load_sessions_config()
    if not cfg.get("enabled", False):
        return None
    try:
        from .session_manager import SessionManager
        project_root = Path(__file__).parent.parent.parent
        db_path = (project_root / cfg.get("db_path", "sessions/checkpoints.db")).resolve()
        key = str(db_path)
        if key not in _SM_REGISTRY:
            registry_path = project_root / cfg.get("registry_path", "sessions/sessions.json")
            _SM_REGISTRY[key] = SessionManager(db_path=db_path, registry_path=registry_path)
        return _SM_REGISTRY[key]
    except Exception:
        return None


class AgentConfig:
    """Базовый класс для конфигурации агента.

    Наследуйтесь от этого класса и определите:
    - states: список состояний State (с transitions внутри)
    - entry_point: имя начального состояния

    Пример:
        class MyAgent(AgentConfig):
            entry_point = "work"
            states = [
                State(name="work", tools=["calculator"], prompt="...",
                      transitions=["summarize"]),
                State(name="summarize", tools=["summarize"], prompt="...",
                      transitions=["END"]),
            ]

        agent = MyAgent(llm, tools_dict)
        result = agent.invoke(["Посчитай 2+2"])

    Сессии и чекпоинты:
        # Запустить с новой сессией (session_id создаётся автоматически)
        result = agent.invoke(["Задача..."])

        # Продолжить существующую (добавить сообщение)
        result = agent.invoke(["Уточнение..."], session_id=existing_id)

        # Восстановить после падения
        result = agent.resume(session_id)

        # Тегировать чекпоинт
        agent.tag_checkpoint(session_id, "baseline_v1")

        # Форк с правками для тестирования
        new_sid = agent.fork("baseline_v1", edits={"memory": {"key": "val"}})
        result = agent.invoke(["Повтори"], session_id=new_sid)
    """

    states: list[State] = []
    entry_point: str | None = None
    sub_agents: list[str] | tuple[str, ...] | None = None

    def __init__(self, llm, tools_dict: Dict[str, Any], agent_id: str | None = None):
        self.llm = llm
        self.tools_dict = tools_dict
        self.agent_id = agent_id or f"{self.__class__.__name__}_{id(self)}"
        self.agent_name = self.__class__.__module__.split(".")[-2]
        self._graph = None

        if not self.states:
            raise ValueError(
                f"{self.__class__.__name__}: Не определены состояния. "
                f"Установите атрибут класса 'states'"
            )
        if not self.entry_point:
            raise ValueError(
                f"{self.__class__.__name__}: Не определена точка входа. "
                f"Установите атрибут класса 'entry_point'"
            )

        # Инициализация SessionManager и прунинг
        self._sm = _make_session_manager()
        self.last_session_id: str | None = None  # session_id последнего invoke/resume
        if self._sm is not None:
            cfg = _load_sessions_config()
            max_sessions = cfg.get("max_sessions_per_agent", 20)
            self._sm.prune_on_load(self.agent_name, max_sessions)

    def build(self):
        builder = AgentGraphBuilder(self.llm, self.tools_dict)
        builder.register_sub_agents(
            owner_agent_name=self.agent_name,
            sub_agents=getattr(self, "sub_agents", None),
        )
        builder.add_states(self.states)
        builder.set_entry(self.entry_point)
        checkpointer = self._sm.get_checkpointer() if self._sm is not None else None
        self._graph = builder.build(checkpointer=checkpointer)
        return self._graph

    @property
    def graph(self):
        if self._graph is None:
            self.build()
        return self._graph

    # ------------------------------------------------------------------
    # Основной запуск
    # ------------------------------------------------------------------

    def invoke(
        self,
        messages: list[str] | dict,
        session_id: str | None = None,
        config: dict | None = None,
    ) -> dict:
        """Запустить агента.

        Args:
            messages: список строк-сообщений или dict с полным AgentState.
            session_id: ID существующей сессии для продолжения.
                Если None и sessions включены — создаётся новая сессия.
            config: дополнительный конфиг LangGraph (переопределяет defaults).

        Returns:
            Итоговый AgentState.
        """
        if isinstance(messages, dict):
            state = messages
        elif session_id is not None:
            # Продолжение существующей сессии или форк: передаём только новые сообщения.
            # LangGraph сам подтянет memory и summary из последнего checkpoint-а —
            # явная передача {} перетёрла бы сохранённую память.
            state = {"messages": messages}
        else:
            state = {"messages": messages, "memory": {}, "summary": ""}

        callbacks, handler = create_callbacks()
        default_config: dict = {"recursion_limit": 100}
        if callbacks:
            default_config["callbacks"] = callbacks

        # Прокидываем thread_id для чекпоинтера
        if self._sm is not None:
            if session_id is None:
                session_id = self._sm.create_session(self.agent_name)
            default_config.setdefault("configurable", {})["thread_id"] = session_id

        if config:
            default_config.update(config)

        self.last_session_id = session_id
        # Публикуем session_id в контекст потока — call_agent читает его
        # как supervisor_session для crash recovery sub-агентов.
        _token = _active_session_ctx.set(session_id)
        log_run_start(self.agent_id)
        try:
            result = self.graph.invoke(state, config=default_config)
        finally:
            _active_session_ctx.reset(_token)
            log_run_end(self.agent_id, handler)
            if self._sm is not None and session_id:
                self._sm.update_session_meta(session_id, last_state="END")

        return result

    def stream(self, messages: list[str] | dict, session_id: str | None = None,
               config: dict | None = None):
        if isinstance(messages, dict):
            state = messages
        else:
            state = {"messages": messages, "memory": {}, "summary": ""}

        callbacks, handler = create_callbacks()
        default_config: dict = {"recursion_limit": 100}
        if callbacks:
            default_config["callbacks"] = callbacks

        if self._sm is not None:
            if session_id is None:
                session_id = self._sm.create_session(self.agent_name)
            default_config.setdefault("configurable", {})["thread_id"] = session_id

        if config:
            default_config.update(config)

        self.last_session_id = session_id
        _token = _active_session_ctx.set(session_id)
        log_run_start(self.agent_id)
        try:
            yield from self.graph.stream(state, config=default_config)
        finally:
            _active_session_ctx.reset(_token)
            log_run_end(self.agent_id, handler)
            if self._sm is not None and session_id:
                self._sm.update_session_meta(session_id, last_state="END")

    # ------------------------------------------------------------------
    # Работа с сессиями и чекпоинтами
    # ------------------------------------------------------------------

    def resume(self, session_id: str) -> dict:
        """Продолжить прерванную сессию (восстановление после краша).

        Проверяет состояние графа перед запуском:
        - Граф прерван (snap.next не пустой): восстанавливает _memory_store
          и продолжает выполнение с прерванного node.
        - Граф завершён (END): бросает NothingToResumeError.
          Крон/скрипт должен поймать это исключение как сигнал «всё в порядке,
          перезапускать не нужно» — без лишних трат токенов.

        Raises:
            NothingToResumeError: сессия уже завершена, восстанавливать нечего.
        """
        self._require_sessions()

        snap_config = {"configurable": {"thread_id": session_id}}
        snap = self.graph.get_state(snap_config)

        if not snap or not snap.next:
            raise NothingToResumeError(session_id)

        self._restore_memory(session_id)

        callbacks, handler = create_callbacks()
        run_config: dict = {
            "recursion_limit": 100,
            "configurable": {"thread_id": session_id},
        }
        if callbacks:
            run_config["callbacks"] = callbacks

        self.last_session_id = session_id
        _token = _active_session_ctx.set(session_id)
        log_run_start(self.agent_id)
        try:
            result = self.graph.invoke(None, config=run_config)
        finally:
            _active_session_ctx.reset(_token)
            log_run_end(self.agent_id, handler)
            self._sm.update_session_meta(session_id, last_state="END")
        return result

    def restore_memory(self, session_id: str, checkpoint_id: str | None = None) -> dict:
        """Восстановить _memory_store из checkpoint-а без запуска агента.

        Используй когда нужно только загрузить состояние памяти:
        - после перезапуска ядра ноутбука
        - для инспекции содержимого конкретного checkpoint-а
        - перед fork() чтобы проверить что будет в памяти

        Args:
            session_id: ID сессии.
            checkpoint_id: конкретный checkpoint (None = последний).

        Returns:
            Словарь memory из checkpoint-а.
        """
        self._require_sessions()
        return self._restore_memory(session_id, checkpoint_id=checkpoint_id)

    def list_checkpoints(self, session_id: str) -> list[dict]:
        """Список всех чекпоинтов сессии от новых к старым.

        Каждая запись:
            checkpoint_id, state_name, timestamp, is_named, name (если есть)
        """
        self._require_sessions()
        config = {"configurable": {"thread_id": session_id}}
        history = list(self.graph.get_state_history(config))

        pinned_ids = {
            cp["checkpoint_id"]: cp["name"]
            for cp in self._sm.list_named_checkpoints(session_id)
        }

        result = []
        for snap in history:
            meta = snap.metadata or {}

            # "input" — LangGraph-чекпоинт ДО первого node (сырой входной стейт).
            # Память и summary там пустые, для пользователя бесполезен.
            # Фильтруем по source, а не по writes.keys(): в ряде версий LangGraph 0.2
            # writes может быть None даже для реальных нод, и тогда все чекпоинты
            # ошибочно маркировались бы как __start__.
            if meta.get("source") == "input":
                continue

            cp_id = snap.config["configurable"].get("checkpoint_id", "")

            # Определяем имя ноды: writes.keys() содержит имена нод которые
            # записали этот checkpoint. Пропускаем внутренние ключи LangGraph.
            writes = meta.get("writes") or {}
            _INTERNAL_KEYS = {"__start__", "__end__", "__copy__", "__root__"}
            real_keys = [k for k in writes if k not in _INTERNAL_KEYS]
            if real_keys:
                state_name = real_keys[0]
            else:
                # writes пустой или None (специфика Command-нод в ряде версий LG)
                # — показываем шаг выполнения как fallback
                state_name = f"step_{meta.get('step', '?')}"

            # created_at — прямой атрибут StateSnapshot, не поле metadata
            timestamp = snap.created_at or meta.get("created_at", "")

            result.append({
                "checkpoint_id": cp_id,
                "state_name": state_name,
                "timestamp": timestamp,
                "next": list(snap.next),
                "is_named": cp_id in pinned_ids,
                "name": pinned_ids.get(cp_id),
            })
        return result

    def get_checkpoint_state(self, checkpoint_ref: str) -> dict:
        """Получить AgentState в точке чекпоинта.

        Args:
            checkpoint_ref: имя именованного чекпоинта или checkpoint_id.
                Для checkpoint_id нужно также передать session_id через
                get_checkpoint_state_by_id(session_id, checkpoint_id).

        Returns:
            dict с ключами: messages, memory, summary (и др. из AgentState).
        """
        self._require_sessions()
        thread_id, checkpoint_id = self._resolve_checkpoint_ref(checkpoint_ref)
        snap = self.graph.get_state(
            {"configurable": {"thread_id": thread_id, "checkpoint_id": checkpoint_id}}
        )
        if snap is None:
            raise ValueError(f"Чекпоинт не найден: '{checkpoint_ref}'")
        return dict(snap.values)

    def get_checkpoint_state_by_id(self, session_id: str, checkpoint_id: str) -> dict:
        """Получить AgentState по session_id + checkpoint_id."""
        self._require_sessions()
        snap = self.graph.get_state(
            {"configurable": {"thread_id": session_id, "checkpoint_id": checkpoint_id}}
        )
        if snap is None:
            raise ValueError(f"Чекпоинт {checkpoint_id} не найден в сессии {session_id}")
        return dict(snap.values)

    def tag_checkpoint(
        self,
        session_id: str,
        name: str,
        note: str = "",
        checkpoint_id: str | None = None,
    ) -> None:
        """Тегировать чекпоинт именем (защищает от авто-удаления).

        Args:
            session_id: ID сессии.
            name: уникальное имя чекпоинта.
            note: опциональная заметка.
            checkpoint_id: конкретный checkpoint_id. Если None — берётся последний.
        """
        self._require_sessions()
        if checkpoint_id is None:
            snap = self.graph.get_state({"configurable": {"thread_id": session_id}})
            if snap is None:
                raise ValueError(f"Нет чекпоинтов для сессии {session_id}")
            checkpoint_id = snap.config["configurable"]["checkpoint_id"]
        self._sm.save_named_checkpoint(session_id, name, checkpoint_id, note=note)

    def fork(
        self,
        checkpoint_ref: str,
        edits: dict | None = None,
        description: str | None = None,
        checkpoint_id: str | None = None,
    ) -> str:
        """Создать новую независимую сессию из чекпоинта.

        Форк создаёт новый thread_id, копируя состояние из checkpoint_ref
        с опциональными правками. Оригинальная сессия не изменяется.

        Два способа указать источник:
        1. По имени тега (именованный чекпоинт):
               fork("baseline_v1")
        2. По session_id + checkpoint_id (любой авто-чекпоинт из виджета/кода):
               fork(session_id, checkpoint_id="abc123...")

        Args:
            checkpoint_ref: имя именованного чекпоинта ИЛИ session_id если
                            передан аргумент checkpoint_id.
            edits: правки к AgentState, например {"memory": {"key": "new_val"}}.
            description: описание новой сессии.
            checkpoint_id: конкретный checkpoint_id внутри session_id.
                           Если передан — checkpoint_ref трактуется как session_id.

        Returns:
            session_id новой сессии. Запустите агента через invoke(messages, session_id=...).
        """
        self._require_sessions()
        if checkpoint_id is not None:
            # Явно передан checkpoint_id → checkpoint_ref является session_id
            thread_id = checkpoint_ref
        else:
            thread_id, checkpoint_id = self._resolve_checkpoint_ref(checkpoint_ref)

        # Получаем снимок состояния в точке чекпоинта
        source_snap = self.graph.get_state(
            {"configurable": {"thread_id": thread_id, "checkpoint_id": checkpoint_id}}
        )
        if source_snap is None:
            raise ValueError(f"Чекпоинт '{checkpoint_ref}' не найден в базе данных")

        # Создаём новую сессию
        fork_desc = description or f"fork от '{checkpoint_ref}'"
        new_session_id = self._sm.create_session(self.agent_name, description=fork_desc)

        # Строим состояние для форка (snapshot + правки)
        fork_values: dict = dict(source_snap.values)
        if edits:
            for key, value in edits.items():
                if key == "memory" and isinstance(value, dict) and isinstance(fork_values.get("memory"), dict):
                    # Для memory делаем merge, а не полную замену
                    fork_values["memory"] = {**fork_values["memory"], **value}
                else:
                    fork_values[key] = value

        # Определяем последний выполненный node — для корректного routing в новом thread
        writes = source_snap.metadata.get("writes") or {}
        last_ran_node = next(iter(writes.keys()), None)

        # Создаём checkpoint в новом thread через update_state
        # as_node задаёт "какой node только что выполнился" → LangGraph выставит
        # правильный next из snapshot.next
        new_thread_config = {"configurable": {"thread_id": new_session_id}}
        self.graph.update_state(
            new_thread_config,
            values=fork_values,
            as_node=last_ran_node,
        )

        return new_session_id

    # ------------------------------------------------------------------
    # Вспомогательные методы
    # ------------------------------------------------------------------

    def _require_sessions(self) -> None:
        if self._sm is None:
            raise RuntimeError(
                "Sessions не включены. Установите sessions.enabled: true в config.yaml"
            )

    def _resolve_checkpoint_ref(self, ref: str) -> tuple[str, str]:
        """Разрешить имя или checkpoint_id → (thread_id, checkpoint_id)."""
        named = self._sm.get_named_checkpoint(ref)
        if named:
            return named["thread_id"], named["checkpoint_id"]
        raise ValueError(
            f"Именованный чекпоинт '{ref}' не найден. "
            "Сначала создайте его через tag_checkpoint()."
        )

    def _restore_memory(self, session_id: str, checkpoint_id: str | None = None) -> dict:
        """Синхронизировать глобальный _memory_store из checkpoint-а.

        Args:
            session_id: ID сессии.
            checkpoint_id: конкретный checkpoint (None = последний).

        Returns:
            Словарь memory из checkpoint-а (пустой если не найден).
        """
        try:
            from src.tools.tools import _memory_store
            cfg: dict = {"configurable": {"thread_id": session_id}}
            if checkpoint_id:
                cfg["configurable"]["checkpoint_id"] = checkpoint_id
            snap = self.graph.get_state(cfg)
            memory = (snap.values.get("memory") or {}) if snap else {}
            _memory_store.clear()
            _memory_store.update(memory)
            return dict(memory)
        except Exception:
            return {}

    def visualize(self) -> str:
        builder = AgentGraphBuilder(self.llm, self.tools_dict)
        builder.add_states(self.states)
        builder.set_entry(self.entry_point)
        return builder.visualize()

    def __repr__(self) -> str:
        sessions_status = "enabled" if self._sm is not None else "disabled"
        return (
            f"{self.__class__.__name__}("
            f"id={self.agent_id}, "
            f"states={len(self.states)}, "
            f"sessions={sessions_status})"
        )
