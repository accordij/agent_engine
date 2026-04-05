"""Базовый класс для конфигурации агентов."""
from __future__ import annotations

from pathlib import Path
from typing import Any, Dict

from .graph_builder import AgentGraphBuilder
from .state import State
from .logging_utils import create_callbacks, log_run_start, log_run_end


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


def _make_session_manager():
    """Создаёт SessionManager из config.yaml или возвращает None."""
    cfg = _load_sessions_config()
    if not cfg.get("enabled", False):
        return None
    try:
        from .session_manager import SessionManager
        project_root = Path(__file__).parent.parent.parent
        db_path = project_root / cfg.get("db_path", "sessions/checkpoints.db")
        registry_path = project_root / cfg.get("registry_path", "sessions/sessions.json")
        return SessionManager(db_path=db_path, registry_path=registry_path)
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
        log_run_start(self.agent_id)
        try:
            result = self.graph.invoke(state, config=default_config)
        finally:
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

        log_run_start(self.agent_id)
        try:
            yield from self.graph.stream(state, config=default_config)
        finally:
            log_run_end(self.agent_id, handler)
            if self._sm is not None and session_id:
                self._sm.update_session_meta(session_id, last_state="END")

    # ------------------------------------------------------------------
    # Работа с сессиями и чекпоинтами
    # ------------------------------------------------------------------

    def resume(self, session_id: str) -> dict:
        """Продолжить прерванную сессию (восстановление после падения).

        Загружает _memory_store из последнего чекпоинта и продолжает
        граф с того node, на котором остановился агент.
        """
        self._require_sessions()
        self._restore_memory(session_id)

        callbacks, handler = create_callbacks()
        run_config: dict = {
            "recursion_limit": 100,
            "configurable": {"thread_id": session_id},
        }
        if callbacks:
            run_config["callbacks"] = callbacks

        log_run_start(self.agent_id)
        try:
            result = self.graph.invoke(None, config=run_config)
        finally:
            log_run_end(self.agent_id, handler)
            self._sm.update_session_meta(session_id, last_state="END")
        return result

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
            cp_id = snap.config["configurable"].get("checkpoint_id", "")
            # Имя node, которая создала этот checkpoint
            writes = snap.metadata.get("writes") or {}
            state_name = next(iter(writes.keys()), "__start__")
            result.append({
                "checkpoint_id": cp_id,
                "state_name": state_name,
                "timestamp": snap.metadata.get("created_at", ""),
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
    ) -> str:
        """Создать новую независимую сессию из именованного чекпоинта.

        Форк создаёт новый thread_id, копируя состояние из checkpoint_ref
        с опциональными правками. Оригинальная сессия не изменяется.

        Args:
            checkpoint_ref: имя именованного чекпоинта.
            edits: правки к AgentState, например {"memory": {"key": "new_val"}}.
            description: описание новой сессии (по умолчанию "fork от <name>").

        Returns:
            session_id новой сессии. Запустите агента через invoke(messages, session_id=...).
        """
        self._require_sessions()
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

    def _restore_memory(self, session_id: str) -> None:
        """Синхронизировать глобальный _memory_store из последнего чекпоинта."""
        try:
            from src.tools.tools import _memory_store
            snap = self.graph.get_state({"configurable": {"thread_id": session_id}})
            if snap and snap.values.get("memory"):
                _memory_store.clear()
                _memory_store.update(snap.values["memory"])
        except Exception:
            pass

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
