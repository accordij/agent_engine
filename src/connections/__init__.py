"""Пакет подключений к LLM."""
from .clients import get_llm_client, get_gigachat_client, get_lmstudio_client, GigaChatDelayed

__all__ = ["get_llm_client", "get_gigachat_client", "get_lmstudio_client", "GigaChatDelayed"]
