"""Модуль подключений к различным LLM."""
import os
from typing import List, Optional, Any
from time import perf_counter, sleep
from langchain_gigachat.chat_models import GigaChat
from langchain_core.messages import BaseMessage
from langchain_core.callbacks import CallbackManagerForLLMRun
from langchain_core.outputs import ChatResult
from langchain_core.language_models.chat_models import generate_from_stream
from langchain_gigachat.chat_models.gigachat import trim_content_to_stop_sequence
from langchain_openai import ChatOpenAI


# Константы для GigaChat rate limiting
GIGA_DELAY = 6
GIGA_LAST_INVOKE = 0


class GigaChatDelayed(GigaChat):
    """GigaChat с учетом rate limiter (задержка между запросами)."""

    def _generate(
        self,
        messages: List[BaseMessage],
        stop: Optional[List[str]] = None,
        run_manager: Optional[CallbackManagerForLLMRun] = None,
        stream: Optional[bool] = None,
        **kwargs: Any,
    ) -> ChatResult:
        global GIGA_LAST_INVOKE

        # Ожидание для соблюдения rate limit
        if perf_counter() - GIGA_LAST_INVOKE >= GIGA_DELAY:
            pass
        else:
            sleep(GIGA_DELAY - (perf_counter() - GIGA_LAST_INVOKE))

        GIGA_LAST_INVOKE = perf_counter()

        should_stream = stream if stream is not None else self.streaming
        if should_stream:
            stream_iter = self._stream(
                messages, stop=stop, run_manager=run_manager, **kwargs
            )
            return generate_from_stream(stream_iter)

        payload = self._build_payload(messages, **kwargs)
        response = self._client.chat(payload)
        for choice in response.choices:
            trimmed_content = trim_content_to_stop_sequence(
                choice.message.content, stop
            )
            if isinstance(trimmed_content, str):
                choice.message.content = trimmed_content
                break

        return self._create_chat_result(response)


def get_gigachat_client(config: dict):
    """
    Создает клиент GigaChat на основе конфигурации.

    Args:
        config: словарь с настройками GigaChat из config.yaml

    Returns:
        GigaChatDelayed: клиент для работы с GigaChat
    """
    return GigaChatDelayed(
        base_url=os.getenv(config["env_vars"]["base_url"]),
        access_token=os.getenv(config["env_vars"]["access_token"]),
        model=config["model"],
        temperature=config["temperature"],
        timeout=config["timeout"]
    )


def get_lmstudio_client(config: dict):
    """
    Создает клиент LM Studio (OpenAI-совместимый) на основе конфигурации.

    Args:
        config: словарь с настройками LM Studio из config.yaml

    Returns:
        ChatOpenAI: клиент для работы с LM Studio
    """
    return ChatOpenAI(
        base_url=config["base_url"],
        model=config["model"],
        temperature=config["temperature"],
        timeout=config["timeout"],
        api_key="not-needed"  # LM Studio не требует API ключ
    )


def get_llm_client(backend: str, config: dict):
    """
    Фабричная функция для получения LLM клиента.

    Args:
        backend: тип бэкенда ("gigachat" или "lmstudio")
        config: полная конфигурация из config.yaml

    Returns:
        LLM клиент (GigaChatDelayed или ChatOpenAI)
    """
    if backend == "gigachat":
        return get_gigachat_client(config["backends"]["gigachat"])
    elif backend == "lmstudio":
        return get_lmstudio_client(config["backends"]["lmstudio"])
    else:
        raise ValueError(f"Неизвестный бэкенд: {backend}. Используйте 'gigachat' или 'lmstudio'.")
