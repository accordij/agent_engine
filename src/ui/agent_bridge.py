"""AgentBridge — мост между агентным движком и Streamlit UI.

Запускает агента в фоновом daemon-потоке и предоставляет:
- лог событий (tool_start, ai_message, state_transition, ...)
- перехват ui_print() из инструментов агента
- механизм вопрос/ответ для инструмента ask_human
- управление жизненным циклом (старт, стоп, перезапуск)
"""
from __future__ import annotations

import ctypes
import threading
from typing import Any

from src.agents import build_agent, list_agents
from src.agent_engine.logging_utils import set_ui_event_emitter, clear_ui_event_emitter
from src.tools.tools import (
    set_human_input_handler, clear_human_input_handler,
    set_ui_print_handler, clear_ui_print_handler,
    set_ui_image_handler, clear_ui_image_handler,
)


class StopAgentException(Exception):
    """Бросается в ask_human когда пользователь нажал Стоп."""


class AgentBridge:
    """Управляет запуском агента в фоновом потоке и общением с UI."""

    def __init__(self) -> None:
        self._thread: threading.Thread | None = None
        self._stop_event = threading.Event()
        self._events: list[dict[str, Any]] = []
        self._events_lock = threading.Lock()
        self._pending_question: tuple | None = None  # (question, answer_event, answer_holder)
        self._running = False
        self._error: str | None = None
        self.last_session_id: str | None = None  # session_id последнего запуска
        # Счётчик поколений: предотвращает сброс _running старым потоком
        # после того как новый уже запустился.
        self._generation = 0

    # ------------------------------------------------------------------
    # Публичный API
    # ------------------------------------------------------------------

    def start(self, agent_name: str, start_message: str, llm: Any,
              recursion_limit: int = 100,
              session_id: str | None = None) -> None:
        """Остановить предыдущий запуск (если был) и запустить агента заново.

        Args:
            session_id: если передан — продолжает существующую сессию;
                        None — создаётся новая сессия автоматически.
        """
        self._do_stop()
        self._stop_event.clear()
        with self._events_lock:
            self._events.clear()
        self._pending_question = None
        self._error = None
        self._generation += 1
        self._running = True

        set_ui_event_emitter(self._handle_event)
        set_human_input_handler(self._ask_human_ui)
        set_ui_print_handler(self._handle_ui_print)
        set_ui_image_handler(self._handle_ui_image)

        self._thread = threading.Thread(
            target=self._run,
            args=(agent_name, start_message, llm, recursion_limit, self._generation, session_id),
            daemon=True,
        )
        self._thread.start()

    def stop(self) -> None:
        """Запросить остановку агента."""
        self._do_stop()

    def send_answer(self, answer: str) -> None:
        """Передать ответ пользователя ожидающему агенту."""
        pq = self._pending_question
        if pq is not None:
            question, answer_event, answer_holder = pq
            answer_holder[0] = answer
            answer_event.set()

    @property
    def is_running(self) -> bool:
        return self._running

    @property
    def pending_question(self) -> str | None:
        pq = self._pending_question
        return pq[0] if pq is not None else None

    @property
    def events(self) -> list[dict[str, Any]]:
        with self._events_lock:
            return list(self._events)

    @property
    def error(self) -> str | None:
        return self._error

    @staticmethod
    def available_agents() -> list[str]:
        return list_agents()

    # ------------------------------------------------------------------
    # Внутренние методы
    # ------------------------------------------------------------------

    def _run(self, agent_name: str, start_message: str, llm: Any,
             recursion_limit: int, generation: int,
             session_id: str | None = None) -> None:
        try:
            agent = build_agent(agent_name, llm)
            agent.invoke(
                [start_message],
                session_id=session_id,
                config={"recursion_limit": recursion_limit},
            )
            # Сохраняем session_id последнего запуска для UI
            self.last_session_id = agent.last_session_id
        except (StopAgentException, KeyboardInterrupt):
            self._handle_event({"type": "stopped", "message": "Агент остановлен пользователем"})
        except Exception as exc:
            self._error = str(exc)
            self._handle_event({"type": "error", "message": str(exc)})
        finally:
            # Сбрасываем _running только если это актуальное поколение потока.
            # Без этой проверки старый поток мог сбросить флаг у нового агента.
            if self._generation == generation:
                self._running = False
                self._pending_question = None
                clear_ui_event_emitter()
                clear_human_input_handler()
                clear_ui_print_handler()
                clear_ui_image_handler()

    def _handle_event(self, event: dict[str, Any]) -> None:
        with self._events_lock:
            self._events.append(event)

    def _handle_ui_print(self, text: str) -> None:
        self._handle_event({"type": "print", "text": text})

    def _handle_ui_image(self, path: str, caption: str) -> None:
        self._handle_event({"type": "image", "path": path, "caption": caption})

    def _ask_human_ui(self, question: str) -> str:
        """Блокирует поток агента до получения ответа из UI."""
        answer_event = threading.Event()
        answer_holder: list[str | None] = [None]
        self._pending_question = (question, answer_event, answer_holder)

        while not answer_event.is_set():
            if self._stop_event.is_set():
                self._pending_question = None
                raise StopAgentException()
            answer_event.wait(timeout=0.3)

        self._pending_question = None
        return answer_holder[0] or ""

    def _do_stop(self) -> None:
        self._stop_event.set()
        # Разблокировать ask_human если он ждёт ответа
        pq = self._pending_question
        if pq is not None:
            _, answer_event, answer_holder = pq
            answer_holder[0] = ""
            answer_event.set()
        self._running = False
        # Принудительно бросить KeyboardInterrupt в поток агента.
        # Без этого агент продолжает работать если он не вызывает ask_human
        # (например зациклился в LLM-вызовах без ожидания ввода пользователя).
        self._force_raise_in_thread()

    def _force_raise_in_thread(self) -> None:
        """Бросает KeyboardInterrupt в поток агента через CPython API."""
        thread = self._thread
        if thread is None or not thread.is_alive():
            return
        res = ctypes.pythonapi.PyThreadState_SetAsyncExc(
            ctypes.c_ulong(thread.ident),
            ctypes.py_object(KeyboardInterrupt),
        )
        if res > 1:
            # Откат если зацепило несколько потоков (не должно случаться)
            ctypes.pythonapi.PyThreadState_SetAsyncExc(
                ctypes.c_ulong(thread.ident), None
            )
