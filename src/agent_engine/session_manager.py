"""Менеджер сессий агентов: реестр, именованные чекпоинты, прунинг."""
import json
import sqlite3
import uuid
from datetime import datetime
from pathlib import Path
from typing import Any

from langgraph.checkpoint.sqlite import SqliteSaver

class SessionManager:
    """Управляет реестром сессий и именованными чекпоинтами агентов.

    Хранит две вещи:
    - checkpoints.db  — LangGraph SqliteSaver со всем AgentState на каждом шаге
    - sessions.json   — наш реестр: метаданные сессий и именованные чекпоинты
    """

    def __init__(self, db_path: str | Path, registry_path: str | Path) -> None:
        self.db_path = Path(db_path)
        self.registry_path = Path(registry_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self.registry_path.parent.mkdir(parents=True, exist_ok=True)
        self._checkpointer: SqliteSaver | None = None
        self._conn: sqlite3.Connection | None = None

    # ------------------------------------------------------------------
    # Checkpointer (singleton)
    # ------------------------------------------------------------------

    def get_checkpointer(self) -> SqliteSaver:
        """Вернуть singleton SqliteSaver.

        Создаём соединение вручную через sqlite3.connect() — это позволяет
        избежать проблемы с from_conn_string(), который возвращает context manager,
        а не готовый объект. check_same_thread=False нужен потому что агент
        работает в отдельном потоке (особенно в Streamlit).
        """
        if self._checkpointer is None:
            self._conn = sqlite3.connect(str(self.db_path), check_same_thread=False, timeout=30)
            self._checkpointer = SqliteSaver(self._conn)
        return self._checkpointer

    def close(self) -> None:
        """Закрыть соединение с БД чекпоинтов."""
        if self._conn is not None:
            try:
                self._conn.close()
            except Exception:
                pass
            self._conn = None
            self._checkpointer = None

    # ------------------------------------------------------------------
    # Реестр (чтение / запись)
    # ------------------------------------------------------------------

    def _load(self) -> dict:
        if not self.registry_path.exists():
            return {"sessions": {}, "named_checkpoints": {}, "sub_session_links": {}}
        data = json.load(open(self.registry_path, encoding="utf-8"))
        data.setdefault("sub_session_links", {})
        return data

    def _save(self, data: dict) -> None:
        with open(self.registry_path, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)

    # ------------------------------------------------------------------
    # Сессии
    # ------------------------------------------------------------------

    def create_session(self, agent_name: str, description: str = "") -> str:
        """Зарегистрировать новую сессию, вернуть session_id (= thread_id)."""
        session_id = str(uuid.uuid4())
        data = self._load()
        data["sessions"][session_id] = {
            "agent": agent_name,
            "created_at": datetime.now().isoformat(),
            "updated_at": datetime.now().isoformat(),
            "description": description,
            "last_state": None,
        }
        self._save(data)
        return session_id

    def update_session_meta(self, session_id: str, **kwargs: Any) -> None:
        """Обновить метаданные сессии (last_state, description и т.д.)."""
        data = self._load()
        if session_id not in data["sessions"]:
            return
        data["sessions"][session_id].update(kwargs)
        data["sessions"][session_id]["updated_at"] = datetime.now().isoformat()
        self._save(data)

    def list_sessions(self, agent_name: str | None = None) -> list[dict]:
        """Список сессий, отсортированный от новых к старым."""
        data = self._load()
        result = []
        for sid, meta in data["sessions"].items():
            if agent_name is None or meta.get("agent") == agent_name:
                result.append({"session_id": sid, **meta})
        return sorted(result, key=lambda x: x.get("updated_at", ""), reverse=True)

    def get_last_session(self, agent_name: str) -> dict | None:
        """Вернуть последнюю сессию для агента или None."""
        sessions = self.list_sessions(agent_name)
        return sessions[0] if sessions else None

    def delete_session(self, session_id: str) -> None:
        """Удалить сессию из реестра и все её чекпоинты из SQLite."""
        data = self._load()
        data["sessions"].pop(session_id, None)
        # Именованные чекпоинты этой сессии тоже удаляем из реестра
        orphaned = [
            name for name, cp in data["named_checkpoints"].items()
            if cp.get("thread_id") == session_id
        ]
        for name in orphaned:
            data["named_checkpoints"].pop(name)
        self._save(data)
        self._delete_thread_from_sqlite(session_id)

    # ------------------------------------------------------------------
    # Именованные чекпоинты
    # ------------------------------------------------------------------

    def save_named_checkpoint(
        self,
        session_id: str,
        name: str,
        checkpoint_id: str,
        note: str = "",
    ) -> None:
        """Сохранить именованный (защищённый) чекпоинт."""
        data = self._load()
        data["named_checkpoints"][name] = {
            "thread_id": session_id,
            "checkpoint_id": checkpoint_id,
            "created_at": datetime.now().isoformat(),
            "note": note,
        }
        self._save(data)

    def get_named_checkpoint(self, name: str) -> dict | None:
        """Получить именованный чекпоинт по имени."""
        return self._load()["named_checkpoints"].get(name)

    def list_named_checkpoints(self, session_id: str | None = None) -> list[dict]:
        """Список именованных чекпоинтов (для сессии или всех)."""
        data = self._load()
        result = []
        for name, cp in data["named_checkpoints"].items():
            if session_id is None or cp.get("thread_id") == session_id:
                result.append({"name": name, **cp})
        return sorted(result, key=lambda x: x.get("created_at", ""), reverse=True)

    def rename_checkpoint(self, old_name: str, new_name: str) -> None:
        """Переименовать именованный чекпоинт."""
        data = self._load()
        if old_name not in data["named_checkpoints"]:
            raise KeyError(f"Чекпоинт '{old_name}' не найден")
        if new_name in data["named_checkpoints"]:
            raise ValueError(f"Имя '{new_name}' уже занято")
        data["named_checkpoints"][new_name] = data["named_checkpoints"].pop(old_name)
        self._save(data)

    def delete_named_checkpoint(self, name: str) -> None:
        """Удалить именованный чекпоинт из реестра.

        Сам checkpoint в SQLite не трогаем — он принадлежит сессии
        и будет удалён вместе с ней при прунинге или ручном удалении.
        """
        data = self._load()
        data["named_checkpoints"].pop(name, None)
        self._save(data)

    def get_pinned_checkpoint_ids(self) -> set[str]:
        """Множество checkpoint_id всех именованных чекпоинтов."""
        data = self._load()
        return {cp["checkpoint_id"] for cp in data["named_checkpoints"].values()}

    def get_protected_session_ids(self) -> set[str]:
        """Множество session_id, у которых есть хотя бы один именованный чекпоинт."""
        data = self._load()
        return {cp["thread_id"] for cp in data["named_checkpoints"].values()}

    # ------------------------------------------------------------------
    # Прунинг — вызывается при загрузке агента
    # ------------------------------------------------------------------

    def prune_on_load(self, agent_name: str, max_sessions: int) -> int:
        """Удалить старые сессии если превышен лимит.

        Сессии с именованными чекпоинтами защищены.
        Возвращает число удалённых сессий.
        """
        if max_sessions <= 0:
            return 0
        sessions = self.list_sessions(agent_name)
        if len(sessions) <= max_sessions:
            return 0

        protected = self.get_protected_session_ids()
        deleted = 0
        # Сессии уже отсортированы от новых к старым — удаляем с хвоста
        for session in sessions[max_sessions:]:
            sid = session["session_id"]
            if sid in protected:
                continue
            self.delete_session(sid)
            deleted += 1
        return deleted

    # ------------------------------------------------------------------
    # Ссылки supervisor → sub-agent session (для crash recovery)
    # ------------------------------------------------------------------

    def link_sub_session(
        self, supervisor_session_id: str, agent_name: str, sub_session_id: str
    ) -> None:
        """Записать на диск: supervisor X при вызове agent_name использовал sub_session Y.

        Вызывается сразу до запуска sub-агента, до любого checkpoint-а.
        Это позволяет найти сессию sub-агента при resume supervisor-а,
        даже если supervisor упал до того как его node завершился.
        """
        data = self._load()
        key = f"{supervisor_session_id}::{agent_name}"
        data["sub_session_links"][key] = {
            "sub_session_id": sub_session_id,
            "created_at": datetime.now().isoformat(),
        }
        self._save(data)

    def get_linked_sub_session(
        self, supervisor_session_id: str, agent_name: str
    ) -> str | None:
        """Найти session_id sub-агента по supervisor + agent_name."""
        data = self._load()
        key = f"{supervisor_session_id}::{agent_name}"
        entry = data["sub_session_links"].get(key)
        return entry["sub_session_id"] if entry else None

    def clear_sub_session_link(
        self, supervisor_session_id: str, agent_name: str
    ) -> None:
        """Удалить ссылку после успешного завершения sub-агента."""
        data = self._load()
        key = f"{supervisor_session_id}::{agent_name}"
        data["sub_session_links"].pop(key, None)
        self._save(data)

    # ------------------------------------------------------------------
    # SQLite: прямое удаление чекпоинтов по thread_id
    # ------------------------------------------------------------------

    def _delete_thread_from_sqlite(self, thread_id: str) -> None:
        """Удалить все записи thread_id из LangGraph-таблиц SQLite."""
        if not self.db_path.exists():
            return
        try:
            with sqlite3.connect(str(self.db_path)) as conn:
                for table in ("checkpoint_writes", "checkpoint_blobs", "checkpoints"):
                    try:
                        conn.execute(
                            f"DELETE FROM {table} WHERE thread_id = ?",  # noqa: S608
                            (thread_id,),
                        )
                    except sqlite3.OperationalError:
                        pass  # Таблица ещё не создана
                conn.commit()
        except sqlite3.Error:
            pass
