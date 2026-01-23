"""Утилиты отладочного логирования."""
from pathlib import Path
import yaml


def log_prompts_enabled(config_path: str | Path = "config.yaml") -> bool:
    """Возвращает флаг логирования промптов из config.yaml."""
    try:
        with open(config_path, "r", encoding="utf-8") as file:
            config = yaml.safe_load(file) or {}
        return bool(config.get("debug", {}).get("log_prompts", False))
    except Exception:
        return False
