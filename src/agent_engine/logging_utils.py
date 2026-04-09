"""Утилиты логирования агента: callbacks, rich-форматирование, метрики."""
from __future__ import annotations

import os
import re
import time
import uuid
from datetime import datetime
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Protocol
from uuid import UUID

import yaml
from langchain_core.callbacks import BaseCallbackHandler
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage, ToolMessage
from rich.console import Console
from rich.theme import Theme


_DEFAULT_COLORS = {
    "system": "#8fb3ff",
    "human": "#9ad4ff",
    "assistant": "#a6e3a1",
    "reasoning": "#b4f0b6",
    "tool": "#f9c97b",
    "tool.name": "bold #f6c177",
    "warning": "bold #f5c2a8",
    "error": "bold #f38ba8",
    "state": "bold #cba6f7",
    "info": "#d8dee9",
    "tokens": "#89dceb",
    "memory": "#f5bde6",
    "run": "bold #e5e9f0",
}

_renderer: "Renderer | None" = None
_config: dict = {}
_log_path: Path | None = None
_ROLE_FILTER_KEYS = ("system", "human", "tools", "assistant", "state", "memory")
_DEFAULT_ROLE_FILTERS = {k: True for k in _ROLE_FILTER_KEYS}
_VALID_RENDERERS = {"off", "auto", "rich", "ansi"}


class Renderer(Protocol):
    def print(self, text: str) -> None:
        ...


class RichRenderer:
    def __init__(self, colors_cfg: dict | None = None):
        self._console = Console(
            theme=_build_theme(colors_cfg),
            soft_wrap=True,
            highlight=False,
            force_jupyter=False,
        )

    def print(self, text: str) -> None:
        self._console.print(text)


class AnsiRenderer:
    def __init__(self, colors_cfg: dict | None = None):
        self._console = Console(
            theme=_build_theme(colors_cfg),
            force_terminal=True,
            force_jupyter=False,
            soft_wrap=True,
            highlight=False,
        )

    def print(self, text: str) -> None:
        self._console.print(text)


def _build_theme(colors_cfg: dict | None = None) -> Theme:
    merged = dict(_DEFAULT_COLORS)
    if colors_cfg:
        key_map = {"tool_name": "tool.name"}
        for k, v in colors_cfg.items():
            theme_key = key_map.get(k, k)
            if theme_key in merged:
                merged[theme_key] = v
    return Theme(merged)


def _is_vscode() -> bool:
    term_program = (os.getenv("TERM_PROGRAM") or "").lower()
    return any(
        (
            os.getenv("VSCODE_PID"),
            os.getenv("VSCODE_CWD"),
            os.getenv("VSCODE_VERBOSE_LOGGING"),
            term_program == "vscode",
        )
    )


def _is_vscode_notebook() -> bool:
    if not _is_jupyter():
        return False
    if _is_vscode():
        return True
    term_program = (os.getenv("TERM_PROGRAM") or "").lower()
    return bool(
        os.getenv("VSCODE_PID")
        or os.getenv("VSCODE_CWD")
        or os.getenv("VSCODE_VERBOSE_LOGGING")
        or term_program == "vscode"
    )


def _resolve_renderer_mode() -> str:
    configured = str(_config.get("renderer", "auto")).strip().lower()
    if configured not in _VALID_RENDERERS:
        configured = "auto"
    if configured != "auto":
        return configured
    if _is_jupyter() or _is_vscode_notebook():
        return "ansi"
    return "rich"


def _get_renderer() -> Renderer | None:
    global _renderer
    if _renderer is not None:
        return _renderer

    mode = _resolve_renderer_mode()
    colors_cfg = _config.get("colors", None)

    if mode == "off":
        _renderer = None
    elif mode == "ansi":
        _renderer = AnsiRenderer(colors_cfg)
    elif mode == "rich":
        _renderer = RichRenderer(colors_cfg)
    else:
        _renderer = RichRenderer(colors_cfg)
    return _renderer


def _strip_rich_markup(text: str) -> str:
    # Remove only our Rich style tags, keep debug labels like [SYS]/[USER].
    style_tags = set(_DEFAULT_COLORS.keys())

    def _replace_tag(match: re.Match[str]) -> str:
        raw = match.group(1).strip()
        tag = raw[1:] if raw.startswith("/") else raw
        return "" if tag in style_tags or raw == "/" else match.group(0)

    cleaned = re.sub(r"\[([^\]]+)\]", _replace_tag, text)
    # Unescape literals emitted as \[...\] for Rich so file logs stay readable.
    return cleaned.replace("\\[", "[").replace("\\]", "]")


def _close_log_file() -> None:
    global _log_path
    _log_path = None


def _init_log_file() -> None:
    global _log_path
    _close_log_file()
    logs_dir = Path(_config.get("logs_dir", "logs"))
    logs_dir.mkdir(parents=True, exist_ok=True)
    file_name = _config.get("file_name") or f"agent_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
    _log_path = logs_dir / file_name
    # Создаём файл и сразу закрываем — дальше пишем open/append/close на строку,
    # чтобы не держать блокировку между вызовами агента.
    with _log_path.open("a", encoding="utf-8"):
        pass


def _write_log_line(text: str) -> None:
    if not _log_path:
        return
    plain = _strip_rich_markup(text).strip()
    if not plain:
        return
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    line = f"{timestamp} {plain}\n"
    with _log_path.open("a", encoding="utf-8") as f:
        f.write(line)
        f.flush()


def _is_jupyter() -> bool:
    try:
        from IPython import get_ipython  # type: ignore

        shell = get_ipython()
        if shell is None:
            return False
        return shell.__class__.__name__ == "ZMQInteractiveShell"
    except Exception:
        return False


def _normalize_role(role: str | None) -> str | None:
    if role is None:
        return None
    normalized = role.strip().lower()
    if normalized == "tool":
        return "tools"
    if normalized in _ROLE_FILTER_KEYS:
        return normalized
    return None


def _merged_role_filters(filters: dict | None) -> dict:
    merged = dict(_DEFAULT_ROLE_FILTERS)
    if not isinstance(filters, dict):
        return merged
    for role_key in _ROLE_FILTER_KEYS:
        if role_key in filters:
            merged[role_key] = bool(filters[role_key])
    if "tool" in filters:
        merged["tools"] = bool(filters["tool"])
    return merged


def _is_role_enabled(role: str | None, scope: str) -> bool:
    normalized = _normalize_role(role)
    if normalized is None:
        return True
    filters = (_config.get("filters") or {}).get(scope, {})
    return bool(filters.get(normalized, True))


def _emit(text: str, role: str | None = None) -> None:
    global_allowed = _is_role_enabled(role, "global")
    if global_allowed:
        _write_log_line(text)
    if not global_allowed:
        return
    if _is_jupyter() and not _is_role_enabled(role, "jupyter"):
        return
    renderer = _get_renderer()
    if renderer is None:
        return
    renderer.print(text)


def load_logging_config(config_path: str = "config.yaml") -> dict:
    global _config
    try:
        with open(config_path, "r", encoding="utf-8") as f:
            full = yaml.safe_load(f) or {}
        logging_cfg = full.get("logging", {}) or {}
        # Prefer nested logging.raw_io; keep backward compatibility with legacy top-level key.
        logging_cfg["raw_io_enabled"] = bool(logging_cfg.get("raw_io", full.get("logging_raw_io", False)))
        logging_cfg["aggregated"] = bool(logging_cfg.get("aggregated", False))
        renderer = str(logging_cfg.get("renderer", "auto")).strip().lower()
        logging_cfg["renderer"] = renderer if renderer in _VALID_RENDERERS else "auto"
        filters_cfg = logging_cfg.get("filters", {}) or {}
        global_filters = _merged_role_filters(filters_cfg.get("global", {}))
        jupyter_filters = _merged_role_filters(filters_cfg.get("jupyter", {}))
        logging_cfg["filters"] = {"global": global_filters, "jupyter": jupyter_filters}
        _config = logging_cfg
    except Exception:
        _config = {}
    return _config


def get_level() -> str:
    return _config.get("level", "off")


def is_enabled() -> bool:
    return get_level() != "off"


def is_detailed() -> bool:
    return get_level() == "detailed"


def is_raw_io_enabled() -> bool:
    return bool(_config.get("raw_io_enabled", False))


def is_aggregated_enabled() -> bool:
    return bool(_config.get("aggregated", False))


def _emit_raw_lines(label: str, payload: Any) -> None:
    _emit(f"  [info]{label}[/]")
    lines = str(payload).splitlines() or [""]
    for line in lines:
        _emit(f"    [info]{line}[/]")


class AgentCallbackHandler(BaseCallbackHandler):
    def __init__(self):
        self.total_input_tokens = 0
        self.total_output_tokens = 0
        self.total_tokens = 0
        self.llm_calls = 0
        self.tool_calls = 0
        self._started_llm_calls = 0
        self._current_request_id = 0
        self._pending_message_count = 0
        self._pending_context_delta: list[Any] = []
        self._last_context_signatures: list[str] = []

    def on_chat_model_start(
        self,
        serialized: Dict[str, Any],
        messages: List[List[Any]],
        *,
        run_id: UUID,
        parent_run_id: Optional[UUID] = None,
        tags: Optional[List[str]] = None,
        metadata: Optional[Dict[str, Any]] = None,
        **kwargs: Any,
    ) -> None:
        if not is_enabled():
            return
        msg_count = sum(len(batch) for batch in messages) if messages else 0
        if is_detailed():
            if is_aggregated_enabled():
                flat_messages = [msg for batch in messages for msg in batch] if messages else []
                signatures = [_message_signature(msg) for msg in flat_messages]
                first_diff_idx = _first_diff_index(self._last_context_signatures, signatures)
                self._started_llm_calls += 1
                self._current_request_id = self._started_llm_calls
                self._pending_message_count = msg_count
                self._pending_context_delta = flat_messages[first_diff_idx:]
                self._last_context_signatures = signatures
            else:
                _emit(f"  [tokens]{'=' * 12} LLM call | context: {msg_count} messages {'=' * 12}[/]")
                for batch in messages:
                    for msg in batch:
                        _print_message(msg)
            if is_raw_io_enabled():
                _emit_raw_lines("RAW_REQUEST:", messages)
        else:
            _emit(f"  [tokens]LLM call | {msg_count} msgs[/]")

    def on_llm_end(self, response: Any, *, run_id: UUID, **kwargs: Any) -> None:
        self.llm_calls += 1
        inp, out, total = _extract_token_usage(response)
        self.total_input_tokens += inp
        self.total_output_tokens += out
        self.total_tokens += total
        # Отправить AI-сообщение в UI (только с текстовым содержимым)
        if response.generations:
            for gen_list in response.generations:
                for gen in gen_list:
                    msg = getattr(gen, "message", None)
                    if msg and getattr(msg, "content", None) and str(msg.content).strip():
                        _emit_ui_event({"type": "ai_message", "content": str(msg.content)})
        _emit_ui_event({"type": "stats", "data": self.get_summary()})
        if not is_enabled():
            return
        if is_detailed() and is_aggregated_enabled():
            req_id = self._current_request_id or self.llm_calls
            _emit(f"  [tokens]REQ {req_id} | msgs={self._pending_message_count} in={inp} out={out}[/]")
            for msg in self._pending_context_delta:
                _print_message(msg)
            self._pending_context_delta = []
            self._pending_message_count = 0
            if is_raw_io_enabled():
                _emit_raw_lines("RAW_RESPONSE:", response)
            return
        if total > 0:
            _emit(
                f"  [tokens]{'=' * 12} tokens: {inp} in / {out} out / {total} total (cumul: {self.total_tokens}) {'=' * 12}[/]"
            )
        if is_detailed():
            if response.generations:
                for gen_list in response.generations:
                    for gen in gen_list:
                        msg = getattr(gen, "message", None)
                        if msg:
                            _print_message(msg)
            if is_raw_io_enabled():
                _emit_raw_lines("RAW_RESPONSE:", response)

    def on_llm_error(self, error: BaseException, *, run_id: UUID, **kwargs: Any) -> None:
        _emit_ui_event({"type": "llm_error", "error": str(error)})
        if not is_enabled():
            return
        self._pending_context_delta = []
        self._pending_message_count = 0
        _emit(f"  [error]LLM ERROR: {error}[/]")

    def on_tool_start(
        self,
        serialized: Dict[str, Any],
        input_str: str,
        *,
        run_id: UUID,
        parent_run_id: Optional[UUID] = None,
        tags: Optional[List[str]] = None,
        metadata: Optional[Dict[str, Any]] = None,
        **kwargs: Any,
    ) -> None:
        self.tool_calls += 1
        name = serialized.get("name", "unknown")
        _emit_ui_event({"type": "tool_start", "name": name, "params": input_str})
        if not is_enabled():
            return
        if is_detailed():
            _emit(f"  [tool.name]TOOL {name}[/] [tool]params={input_str}[/]", role="tools")
        else:
            _emit(f"  [tool.name]TOOL[/] {name}", role="tools")

    def on_tool_end(self, output: Any, *, run_id: UUID, **kwargs: Any) -> None:
        _emit_ui_event({"type": "tool_end", "output": str(output)[:300]})
        if not is_detailed():
            return
        _emit(f"  [tool]  -> {str(output)}[/]", role="tools")

    def on_tool_error(self, error: BaseException, *, run_id: UUID, **kwargs: Any) -> None:
        err_name = error.__class__.__name__
        if err_name == "EarlyBreakTransition":
            _emit_ui_event({"type": "tool_break", "message": "EARLY_BREAK"})
            if not is_enabled():
                return
            _emit("  [state]TOOL BREAK[/]", role="tools")
            return
        _emit_ui_event({"type": "tool_error", "error": str(error)})
        if not is_enabled():
            return
        _emit(f"  [error]TOOL ERROR: {error}[/]", role="tools")

    def get_summary(self) -> dict:
        return {
            "llm_calls": self.llm_calls,
            "tool_calls": self.tool_calls,
            "input_tokens": self.total_input_tokens,
            "output_tokens": self.total_output_tokens,
            "total_tokens": self.total_tokens,
        }


def _coerce_token_value(value: Any) -> int:
    if value is None:
        return 0
    if isinstance(value, int):
        return value
    try:
        return int(value)
    except (TypeError, ValueError):
        return 0


def _read_usage_field(usage: Any, *field_names: str) -> int:
    if usage is None:
        return 0
    for field_name in field_names:
        if isinstance(usage, dict):
            raw = usage.get(field_name)
        else:
            raw = getattr(usage, field_name, None)
        value = _coerce_token_value(raw)
        if value:
            return value
    return 0


def _extract_token_usage(response: Any) -> tuple[int, int, int]:
    usage: Any = {}
    if hasattr(response, "llm_output") and response.llm_output:
        usage = response.llm_output.get("token_usage", {})
    if not usage and hasattr(response, "generations"):
        for gen_list in response.generations:
            for gen in gen_list:
                msg = getattr(gen, "message", None)
                if msg and hasattr(msg, "usage_metadata") and msg.usage_metadata:
                    um = msg.usage_metadata
                    inp = _read_usage_field(um, "input_tokens", "prompt_tokens")
                    out = _read_usage_field(um, "output_tokens", "completion_tokens")
                    total = _read_usage_field(um, "total_tokens") or (inp + out)
                    return inp, out, total
                info = getattr(gen, "generation_info", None) or {}
                if "token_usage" in info:
                    usage = info["token_usage"]
                    break
    inp = _read_usage_field(usage, "prompt_tokens", "input_tokens")
    out = _read_usage_field(usage, "completion_tokens", "output_tokens")
    total = _read_usage_field(usage, "total_tokens") or (inp + out)
    return inp, out, total


def _message_signature(msg: Any) -> str:
    if isinstance(msg, SystemMessage):
        return f"system::{str(msg.content)}"
    if isinstance(msg, HumanMessage):
        return f"human::{str(msg.content)}"
    if isinstance(msg, ToolMessage):
        name = getattr(msg, "name", "?")
        return f"tool::{name}::{str(msg.content)}"
    if isinstance(msg, AIMessage):
        return f"assistant::{str(msg.content)}::tool_calls={str(msg.tool_calls or [])}"
    content = getattr(msg, "content", str(msg))
    return f"{msg.__class__.__name__}::{str(content)}"


def _first_diff_index(previous: list[str], current: list[str]) -> int:
    max_common = min(len(previous), len(current))
    for idx in range(max_common):
        if previous[idx] != current[idx]:
            return idx
    return max_common


def _print_message(msg: Any) -> None:
    if isinstance(msg, SystemMessage):
        _emit(f"    [system]\\[SYS] {str(msg.content)}[/]", role="system")
    elif isinstance(msg, HumanMessage):
        _emit(f"    [human]\\[USER] {str(msg.content)}[/]", role="human")
    elif isinstance(msg, ToolMessage):
        name = getattr(msg, "name", "?")
        _emit(f"    [tool]\\[TOOL:{name}] {str(msg.content)}[/]", role="tools")
    elif isinstance(msg, AIMessage):
        if msg.tool_calls:
            for tc in msg.tool_calls:
                _emit(f"    [assistant]\\[AI->TOOL] {tc['name']}({tc.get('args', {})})[/]", role="assistant")
        if msg.content:
            _emit(f"    [assistant]\\[AI] {str(msg.content)}[/]", role="assistant")


_current_run_id: str | None = None
_run_start_time: float = 0

# UI event emitter hook — устанавливается AgentBridge при работе в Streamlit
_ui_event_emitter: Callable[[dict], None] | None = None


def set_ui_event_emitter(fn: Callable[[dict], None]) -> None:
    """Установить UI event emitter (вызывается из AgentBridge)."""
    global _ui_event_emitter
    _ui_event_emitter = fn


def clear_ui_event_emitter() -> None:
    """Очистить UI event emitter."""
    global _ui_event_emitter
    _ui_event_emitter = None


def _emit_ui_event(event: dict) -> None:
    if _ui_event_emitter is not None:
        try:
            _ui_event_emitter(event)
        except Exception:
            pass


def log_run_start(agent_name: str) -> str | None:
    global _current_run_id, _run_start_time
    _current_run_id = uuid.uuid4().hex[:8]
    _run_start_time = time.time()
    _emit_ui_event({"type": "run_start", "agent": agent_name, "run_id": _current_run_id})
    if not is_enabled():
        return _current_run_id
    _emit(f"[run]{'=' * 60}[/]")
    _emit(f"[run]RUN {_current_run_id} | {agent_name} | started[/]")
    _emit(f"[run]{'=' * 60}[/]")
    return _current_run_id


def log_run_end(agent_name: str, handler: AgentCallbackHandler | None = None):
    global _current_run_id
    elapsed = time.time() - _run_start_time
    stats = handler.get_summary() if handler else {}
    _emit_ui_event({"type": "run_end", "agent": agent_name, "elapsed": elapsed, "stats": stats})
    if not is_enabled():
        _current_run_id = None
        return
    _emit(f"[run]{'=' * 60}[/]")
    parts = [f"RUN {_current_run_id} | {agent_name} | {elapsed:.1f}s"]
    if handler:
        s = handler.get_summary()
        parts.append(f"| {s['total_tokens']} tokens ({s['input_tokens']} in / {s['output_tokens']} out)")
        parts.append(f"| {s['llm_calls']} LLM calls, {s['tool_calls']} tool calls")
    _emit(f"[run]{' '.join(parts)}[/]")
    _emit(f"[run]{'=' * 60}[/]")
    _current_run_id = None


def log_state_transition(from_state: str, to_state: str):
    _emit_ui_event({"type": "state_transition", "from": from_state, "to": to_state})
    if not is_enabled():
        return
    _emit(f"[state]STATE {from_state} -> {to_state}[/]", role="state")


def log_memory_snapshot(state_name: str, memory_store: dict, when: str = "exit"):
    if not is_enabled():
        return
    keys = sorted(k for k in memory_store if not k.startswith("__"))
    if not keys:
        _emit(f"  [memory]memory @ {when}({state_name}): empty[/]", role="memory")
        return
    if is_detailed():
        _emit(f"  [memory]memory @ {when}({state_name}):[/]", role="memory")
        for k in keys:
            _emit(f"    [memory]{k}: {str(memory_store[k])}[/]", role="memory")
    else:
        _emit(f"  [memory]memory @ {when}({state_name}): \\[{', '.join(keys)}][/]", role="memory")


def log_warning(message: str):
    _emit_ui_event({"type": "warning", "message": message})
    if not is_enabled():
        return
    _emit(f"  [warning]WARN: {message}[/]")


def log_transition_mode(mode: str, from_state: str, to_state: str):
    _emit_ui_event(
        {
            "type": "transition_mode",
            "mode": mode,
            "from": from_state,
            "to": to_state,
        }
    )
    if not is_enabled():
        return
    _emit(f"  [state]TRANSITION {mode}: {from_state} -> {to_state}[/]", role="state")


def log_reentry(state_name: str):
    if not is_enabled():
        return
    _emit(f"  [warning]REENTRY: recursion limit, re-entering {state_name}[/]")


def init_logging(config_path: str = "config.yaml") -> None:
    global _renderer
    cfg = load_logging_config(config_path)
    _renderer = None
    if not is_enabled():
        _close_log_file()
        return
    _init_log_file()
    mlflow_cfg = cfg.get("mlflow", {})
    if mlflow_cfg.get("enabled", False):
        try:
            import mlflow

            uri = mlflow_cfg.get("tracking_uri", "")
            if uri:
                mlflow.set_tracking_uri(uri)
            mlflow.set_experiment(mlflow_cfg.get("experiment", "agent-runs"))
            mlflow.langchain.autolog()
        except ImportError:
            log_warning("mlflow not installed, skipping")
        except Exception as e:
            log_warning(f"MLflow init failed: {e}")
    _emit(f"[info]logging: level={get_level()}[/]")


def create_callbacks() -> tuple[list, AgentCallbackHandler | None]:
    if not is_enabled():
        return [], None
    handler = AgentCallbackHandler()
    return [handler], handler


__all__ = [
    "AgentCallbackHandler",
    "clear_ui_event_emitter",
    "create_callbacks",
    "get_level",
    "init_logging",
    "is_detailed",
    "is_enabled",
    "load_logging_config",
    "log_memory_snapshot",
    "log_reentry",
    "log_run_end",
    "log_run_start",
    "log_state_transition",
    "log_transition_mode",
    "log_warning",
    "set_ui_event_emitter",
]
