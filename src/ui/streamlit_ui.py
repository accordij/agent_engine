"""Streamlit UI для агентского движка.

Запуск из корня проекта:
    streamlit run src/ui/streamlit_ui.py

Работает независимо от основного кода. Убрать UI = удалить папку src/ui/.
"""
import sys
from pathlib import Path

import streamlit as st
import yaml

# Корень проекта — два уровня вверх от этого файла (src/ui/ -> src/ -> root)
_PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(_PROJECT_ROOT))

from src.connections.clients import get_llm_client
from src.agent_engine.logging_utils import init_logging
from src.ui.agent_bridge import AgentBridge


# ------------------------------------------------------------------
# Инициализация (кэшируется на всё время жизни сервера)
# ------------------------------------------------------------------

@st.cache_resource
def _load_config() -> dict:
    with open(_PROJECT_ROOT / "config.yaml", encoding="utf-8") as f:
        return yaml.safe_load(f)


@st.cache_resource
def _init_llm():
    cfg = _load_config()
    init_logging(str(_PROJECT_ROOT / "config.yaml"))
    return get_llm_client(cfg["active_backend"], cfg)


# ------------------------------------------------------------------
# Session state
# ------------------------------------------------------------------

if "bridge" not in st.session_state:
    st.session_state.bridge = AgentBridge()


# ------------------------------------------------------------------
# Сайдбар — управление (в фрагменте, чтобы кнопки отражали живое состояние)
# ------------------------------------------------------------------

with st.sidebar:
    st.title("Агентский движок")

    # Поля ввода — хранятся в session_state по ключу, доступны из фрагмента
    bridge = st.session_state.bridge
    st.selectbox("Агент", bridge.available_agents(),
                 disabled=bridge.is_running, key="ui_agent_name")
    st.text_area("Стартовое сообщение",
                 placeholder="Опишите задачу для агента...",
                 disabled=bridge.is_running, key="ui_start_message")

    @st.fragment(run_every="1s")
    def sidebar_controls() -> None:
        b: AgentBridge = st.session_state.bridge
        agent_name: str = st.session_state.get("ui_agent_name", "")
        start_message: str = st.session_state.get("ui_start_message", "") or ""

        # Когда агент завершается — делаем полную перерисовку страницы,
        # чтобы selectbox и text_area (вне фрагмента) разблокировались.
        if st.session_state.get("_agent_was_running", False) and not b.is_running:
            st.session_state["_agent_was_running"] = False
            st.rerun(scope="app")
        if b.is_running:
            st.session_state["_agent_was_running"] = True

        col1, col2 = st.columns(2)
        run_clicked = col1.button(
            "▶ Запустить",
            disabled=b.is_running or not start_message.strip(),
            use_container_width=True,
            type="primary",
            key="btn_run",
        )
        stop_clicked = col2.button(
            "⏹ Стоп",
            disabled=not b.is_running,
            use_container_width=True,
            key="btn_stop",
        )

        if run_clicked:
            cfg = _load_config()
            recursion_limit = cfg.get("agent", {}).get("recursion_limit", 100)
            llm = _init_llm()
            b.start(agent_name, start_message.strip(), llm, recursion_limit)
            st.rerun(scope="app")

        if stop_clicked:
            b.stop()
            st.rerun(scope="app")

        st.divider()

        if b.is_running:
            st.info("Работает...")
        elif b.error:
            st.error(f"Ошибка: {b.error}")
        elif b.events:
            st.success("Завершено")
        else:
            st.caption("Выберите агента и введите задачу")

    sidebar_controls()


# ------------------------------------------------------------------
# Основная область — лента событий + вопрос/ответ
# ------------------------------------------------------------------

st.header("Лента событий")


@st.fragment(run_every="0.5s")
def agent_panel() -> None:
    b: AgentBridge = st.session_state.bridge

    # Вопрос от агента — показываем поверх ленты
    pq = b.pending_question
    if pq is not None:
        with st.form("answer_form", clear_on_submit=True):
            st.warning(f"**Вопрос агента:** {pq}")
            answer = st.text_input("Ваш ответ", key="answer_input")
            submitted = st.form_submit_button("Отправить", type="primary")
        if submitted and answer.strip():
            b.send_answer(answer.strip())

    # Лента событий
    events = b.events
    if not events:
        st.caption("Событий пока нет. Запустите агента.")
        return

    for event in events:
        _render_event(event)


def _render_event(event: dict) -> None:
    etype = event.get("type", "")

    if etype == "run_start":
        st.info(f"**Запуск агента** `{event.get('agent')}` (run {event.get('run_id')})")

    elif etype == "run_end":
        elapsed = event.get("elapsed", 0)
        stats = event.get("stats", {})
        tokens = stats.get("total_tokens", 0)
        llm_calls = stats.get("llm_calls", 0)
        tool_calls = stats.get("tool_calls", 0)
        st.success(
            f"**Завершено** за {elapsed:.1f}с — "
            f"{tokens} токенов, {llm_calls} LLM-вызовов, {tool_calls} инструментов"
        )

    elif etype == "ai_message":
        with st.chat_message("assistant"):
            st.markdown(event.get("content", ""))

    elif etype == "tool_start":
        name = event.get("name", "")
        params = event.get("params", "")
        st.markdown(f"🔧 **{name}** `{params}`")

    elif etype == "tool_end":
        output = event.get("output", "")
        st.markdown(f"↩ `{output}`")

    elif etype == "tool_error":
        st.error(f"Ошибка инструмента: {event.get('error')}")

    elif etype == "state_transition":
        st.markdown(
            f"<span style='color:gray'>⟶ состояние: "
            f"**{event.get('from')}** → **{event.get('to')}**</span>",
            unsafe_allow_html=True,
        )

    elif etype == "warning":
        st.warning(event.get("message", ""))

    elif etype == "error":
        st.error(f"**Ошибка:** {event.get('message', '')}")

    elif etype == "stopped":
        st.warning(event.get("message", "Остановлено"))

    elif etype == "llm_error":
        st.error(f"**Ошибка LLM:** {event.get('error', '')}")


agent_panel()
