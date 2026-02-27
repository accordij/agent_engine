"""Утилиты логирования агента: callbacks, rich-форматирование, метрики."""
from __future__ import annotations

import re
import time
import uuid
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, TextIO
from uuid import UUID

import yaml
from langchain_core.callbacks import BaseCallbackHandler
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage, ToolMessage
from rich.console import Console
from rich.theme import Theme


_DEFAULT_COLORS = {
    "system": "bright_blue",
    "human": "cyan",
    "assistant": "green",
    "reasoning": "bright_green",
    "tool": "yellow",
    "tool.name": "bold yellow",
    "warning": "orange3",
    "error": "bold red",
    "state": "bold magenta",
    "info": "dim white",
    "tokens": "dim cyan",
    "memory": "bright_magenta",
    "run": "bold white",
}

_console: Console | None = None
_config: dict = {}
_log_file: TextIO | None = None
_log_path: Path | None = None


def _build_theme(colors_cfg: dict | None = None) -> Theme:
    merged = dict(_DEFAULT_COLORS)
    if colors_cfg:
        key_map = {"tool_name": "tool.name"}
        for k, v in colors_cfg.items():
            theme_key = key_map.get(k, k)
            if theme_key in merged:
                merged[theme_key] = v
    return Theme(merged)


def _get_console() -> Console:
    global _console
    if _console is None:
        colors_cfg = _config.get("colors", None)
        _console = Console(theme=_build_theme(colors_cfg))
    return _console


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
    global _log_file, _log_path
    if _log_file:
        _log_file.close()
    _log_file = None
    _log_path = None


def _init_log_file() -> None:
    global _log_file, _log_path
    _close_log_file()
    logs_dir = Path(_config.get("logs_dir", "logs"))
    logs_dir.mkdir(parents=True, exist_ok=True)
    file_name = _config.get("file_name") or f"agent_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
    _log_path = logs_dir / file_name
    _log_file = _log_path.open("a", encoding="utf-8")


def _write_log_line(text: str) -> None:
    if not _log_file:
        return
    plain = _strip_rich_markup(text).strip()
    if not plain:
        return
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    _log_file.write(f"{timestamp} {plain}\n")
    _log_file.flush()


def _emit(text: str) -> None:
    _get_console().print(text)
    _write_log_line(text)


def load_logging_config(config_path: str = "config.yaml") -> dict:
    global _config
    try:
        with open(config_path, "r", encoding="utf-8") as f:
            full = yaml.safe_load(f) or {}
        logging_cfg = full.get("logging", {}) or {}
        # Prefer nested logging.raw_io; keep backward compatibility with legacy top-level key.
        logging_cfg["raw_io_enabled"] = bool(logging_cfg.get("raw_io", full.get("logging_raw_io", False)))
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
            _emit(f"  [tokens]{'=' * 12} LLM call | context: {msg_count} messages {'=' * 12}[/]")
            for batch in messages:
                for msg in batch:
                    _print_message(msg)
            if is_raw_io_enabled():
                _emit_raw_lines("RAW_REQUEST:", messages)
        else:
            _emit(f"  [tokens]LLM call | {msg_count} msgs[/]")

    def on_llm_end(self, response: Any, *, run_id: UUID, **kwargs: Any) -> None:
        if not is_enabled():
            return
        self.llm_calls += 1
        inp, out, total = _extract_token_usage(response)
        self.total_input_tokens += inp
        self.total_output_tokens += out
        self.total_tokens += total
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
        if not is_enabled():
            return
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
        if not is_enabled():
            return
        self.tool_calls += 1
        name = serialized.get("name", "unknown")
        if is_detailed():
            _emit(f"  [tool.name]TOOL {name}[/] [tool]params={input_str}[/]")
        else:
            _emit(f"  [tool.name]TOOL[/] {name}")

    def on_tool_end(self, output: Any, *, run_id: UUID, **kwargs: Any) -> None:
        if not is_detailed():
            return
        _emit(f"  [tool]  -> {str(output)}[/]")

    def on_tool_error(self, error: BaseException, *, run_id: UUID, **kwargs: Any) -> None:
        if not is_enabled():
            return
        _emit(f"  [error]TOOL ERROR: {error}[/]")

    def get_summary(self) -> dict:
        return {
            "llm_calls": self.llm_calls,
            "tool_calls": self.tool_calls,
            "input_tokens": self.total_input_tokens,
            "output_tokens": self.total_output_tokens,
            "total_tokens": self.total_tokens,
        }


def _extract_token_usage(response: Any) -> tuple[int, int, int]:
    usage: dict = {}
    if hasattr(response, "llm_output") and response.llm_output:
        usage = response.llm_output.get("token_usage", {})
    if not usage and hasattr(response, "generations"):
        for gen_list in response.generations:
            for gen in gen_list:
                msg = getattr(gen, "message", None)
                if msg and hasattr(msg, "usage_metadata") and msg.usage_metadata:
                    um = msg.usage_metadata
                    return (um.get("input_tokens", 0), um.get("output_tokens", 0), um.get("total_tokens", 0))
                info = getattr(gen, "generation_info", None) or {}
                if "token_usage" in info:
                    usage = info["token_usage"]
                    break
    inp = usage.get("prompt_tokens", 0) or usage.get("input_tokens", 0)
    out = usage.get("completion_tokens", 0) or usage.get("output_tokens", 0)
    total = usage.get("total_tokens", 0) or (inp + out)
    return inp, out, total


def _print_message(msg: Any) -> None:
    if isinstance(msg, SystemMessage):
        _emit(f"    [system]\\[SYS] {str(msg.content)}[/]")
    elif isinstance(msg, HumanMessage):
        _emit(f"    [human]\\[USER] {str(msg.content)}[/]")
    elif isinstance(msg, ToolMessage):
        name = getattr(msg, "name", "?")
        _emit(f"    [tool]\\[TOOL:{name}] {str(msg.content)}[/]")
    elif isinstance(msg, AIMessage):
        if msg.tool_calls:
            for tc in msg.tool_calls:
                _emit(f"    [assistant]\\[AI->TOOL] {tc['name']}({tc.get('args', {})})[/]")
        if msg.content:
            _emit(f"    [assistant]\\[AI] {str(msg.content)}[/]")


_current_run_id: str | None = None
_run_start_time: float = 0


def log_run_start(agent_name: str) -> str | None:
    global _current_run_id, _run_start_time
    if not is_enabled():
        return None
    _current_run_id = uuid.uuid4().hex[:8]
    _run_start_time = time.time()
    _emit(f"\n[run]{'=' * 60}[/]")
    _emit(f"[run]RUN {_current_run_id} | {agent_name} | started[/]")
    _emit(f"[run]{'=' * 60}[/]")
    return _current_run_id


def log_run_end(agent_name: str, handler: AgentCallbackHandler | None = None):
    global _current_run_id
    if not is_enabled():
        return
    elapsed = time.time() - _run_start_time
    _emit(f"\n[run]{'=' * 60}[/]")
    parts = [f"RUN {_current_run_id} | {agent_name} | {elapsed:.1f}s"]
    if handler:
        s = handler.get_summary()
        parts.append(f"| {s['total_tokens']} tokens ({s['input_tokens']} in / {s['output_tokens']} out)")
        parts.append(f"| {s['llm_calls']} LLM calls, {s['tool_calls']} tool calls")
    _emit(f"[run]{' '.join(parts)}[/]")
    _emit(f"[run]{'=' * 60}[/]\n")
    _current_run_id = None


def log_state_transition(from_state: str, to_state: str):
    if not is_enabled():
        return
    _emit(f"\n[state]STATE {from_state} -> {to_state}[/]")


def log_memory_snapshot(state_name: str, memory_store: dict, when: str = "exit"):
    if not is_enabled():
        return
    keys = sorted(k for k in memory_store if not k.startswith("__"))
    if not keys:
        _emit(f"  [memory]memory @ {when}({state_name}): empty[/]")
        return
    if is_detailed():
        _emit(f"  [memory]memory @ {when}({state_name}):[/]")
        for k in keys:
            _emit(f"    [memory]{k}: {str(memory_store[k])}[/]")
    else:
        _emit(f"  [memory]memory @ {when}({state_name}): \\[{', '.join(keys)}][/]")


def log_warning(message: str):
    if not is_enabled():
        return
    _emit(f"  [warning]WARN: {message}[/]")


def log_reentry(state_name: str):
    if not is_enabled():
        return
    _emit(f"  [warning]REENTRY: recursion limit, re-entering {state_name}[/]")


def init_logging(config_path: str = "config.yaml") -> None:
    global _console
    cfg = load_logging_config(config_path)
    _console = None
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
    "log_warning",
]
