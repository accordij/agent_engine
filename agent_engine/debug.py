"""Утилиты отладочного логирования."""
from __future__ import annotations

from pathlib import Path
from datetime import datetime
import json
import os
import yaml

_session_id: str | None = None
_session_path: Path | None = None
_event_seq: int = 0
_logging_enabled: bool = False  # Глобальный флаг логирования


def enable_logging() -> None:
    """Включает консольное логирование для отладки."""
    global _logging_enabled
    _logging_enabled = True


def disable_logging() -> None:
    """Отключает консольное логирование."""
    global _logging_enabled
    _logging_enabled = False


def log_prompts_enabled(config_path: str | Path = "config.yaml") -> bool:
    """Возвращает флаг логирования промптов.
    
    Проверяет глобальный флаг _logging_enabled ИЛИ настройку в config.yaml.
    """
    global _logging_enabled
    if _logging_enabled:
        return True
    
    try:
        with open(config_path, "r", encoding="utf-8") as file:
            config = yaml.safe_load(file) or {}
        return bool(config.get("debug", {}).get("log_prompts", False))
    except Exception:
        return False


def _load_config(config_path: str | Path = "config.yaml") -> dict:
    try:
        with open(config_path, "r", encoding="utf-8") as file:
            return yaml.safe_load(file) or {}
    except Exception:
        return {}


def _get_debug_config(config_path: str | Path = "config.yaml") -> dict:
    config = _load_config(config_path)
    debug = config.get("debug", {}) if isinstance(config, dict) else {}
    return {
        "log_sessions": bool(debug.get("log_sessions", False)),
        "log_dir": debug.get("log_dir", "logs"),
        "log_llm_messages": bool(debug.get("log_llm_messages", True)),
        "log_tool_params": bool(debug.get("log_tool_params", True)),
        "log_tool_results": bool(debug.get("log_tool_results", True)),
        "log_state_events": bool(debug.get("log_state_events", True)),
    }


def _sanitize(value):
    if isinstance(value, dict):
        return {str(k): _sanitize(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_sanitize(v) for v in value]
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    try:
        json.dumps(value)
        return value
    except Exception:
        return repr(value)


def _message_to_dict(message) -> dict:
    try:
        from langchain_core.messages import AIMessage, HumanMessage, SystemMessage, ToolMessage
    except Exception:
        return {"type": message.__class__.__name__, "content": repr(message)}

    if isinstance(message, SystemMessage):
        role = "system"
    elif isinstance(message, HumanMessage):
        role = "human"
    elif isinstance(message, ToolMessage):
        role = "tool"
    elif isinstance(message, AIMessage):
        role = "assistant"
    else:
        role = "unknown"

    payload = {
        "role": role,
        "type": message.__class__.__name__,
        "content": _sanitize(getattr(message, "content", None)),
    }

    if isinstance(message, ToolMessage):
        payload["tool_name"] = getattr(message, "name", None)
        payload["tool_call_id"] = getattr(message, "tool_call_id", None)
    if isinstance(message, AIMessage):
        payload["tool_calls"] = _sanitize(getattr(message, "tool_calls", None))
        payload["additional_kwargs"] = _sanitize(getattr(message, "additional_kwargs", None))

    return payload


def serialize_messages(messages: list) -> list[dict]:
    return [_message_to_dict(msg) for msg in messages]


def _init_session(config_path: str | Path = "config.yaml") -> None:
    global _session_id, _session_path, _event_seq
    if _session_path is not None:
        return
    debug = _get_debug_config(config_path)
    if not debug["log_sessions"]:
        return

    log_dir = Path(debug["log_dir"])
    log_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    pid = os.getpid()
    _session_id = f"{timestamp}_{pid}"
    _session_path = log_dir / f"session_{_session_id}.jsonl"
    _event_seq = 0

    _write_event(
        {
            "event": "session_start",
            "data": {
                "log_dir": str(log_dir),
                "config_path": str(config_path),
            },
        }
    )


def _write_event(payload: dict) -> None:
    global _event_seq
    if _session_path is None:
        return
    _event_seq += 1
    payload = {
        "ts": datetime.now().isoformat(timespec="seconds"),
        "session_id": _session_id,
        "seq": _event_seq,
        **payload,
    }
    with _session_path.open("a", encoding="utf-8") as file:
        file.write(json.dumps(_sanitize(payload), ensure_ascii=False) + "\n")


def log_event(
    event_type: str,
    data: dict | None = None,
    *,
    step: int | None = None,
    state: str | None = None,
    config_path: str | Path = "config.yaml",
) -> None:
    debug = _get_debug_config(config_path)
    if not debug["log_sessions"]:
        return

    if event_type in {"llm_request", "llm_response"} and not debug["log_llm_messages"]:
        return
    if event_type == "tool_call" and not debug["log_tool_params"]:
        return
    if event_type == "tool_result" and not debug["log_tool_results"]:
        return
    if event_type.startswith("state_") and not debug["log_state_events"]:
        return

    _init_session(config_path)
    _write_event(
        {
            "event": event_type,
            "step": step,
            "state": state,
            "data": data or {},
        }
    )
