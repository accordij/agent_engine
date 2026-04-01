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

# ------------------------------------------------------------------
# Конфиг читается до первого st.* вызова (нужен для set_page_config)
# ------------------------------------------------------------------

with open(_PROJECT_ROOT / "config.yaml", encoding="utf-8") as _f:
    _startup_cfg = yaml.safe_load(_f)

_st_cfg: dict = _startup_cfg.get("streamlit", {})

st.set_page_config(
    page_title=_st_cfg.get("page_title", "Агентский движок"),
    page_icon=_st_cfg.get("page_icon", "🤖"),
    layout=_st_cfg.get("layout", "wide"),
    initial_sidebar_state=_st_cfg.get("sidebar_state", "expanded"),
)

from src.connections.clients import get_llm_client          # noqa: E402
from src.agent_engine.logging_utils import init_logging     # noqa: E402
from src.ui.agent_bridge import AgentBridge                 # noqa: E402


# ------------------------------------------------------------------
# Темы и шрифты
# ------------------------------------------------------------------

_THEMES: dict[str, dict[str, str]] = {
    "light": {
        "bg":         "#ffffff",
        "sidebar_bg": "#f0f2f6",
        "text":       "#31333f",
        "primary":    "#ff4b4b",
    },
    "dark": {
        "bg":         "#0e1117",
        "sidebar_bg": "#262730",
        "text":       "#fafafa",
        "primary":    "#ff4b4b",
    },
    "catppuccin": {
        "bg":         "#1e1e2e",
        "sidebar_bg": "#181825",
        "text":       "#cdd6f4",
        "primary":    "#a6e3a1",
    },
}

_FONT_PX: dict[str, int] = {"small": 13, "normal": 15, "large": 17, "xlarge": 19}
_CODE_PX: dict[str, int] = {"small": 12, "normal": 13, "large": 14, "xlarge": 15}


def _build_css() -> str:
    c = _THEMES.get(_st_cfg.get("theme", "dark"), _THEMES["dark"])
    fp = _FONT_PX.get(_st_cfg.get("font_size", "normal"), 15)
    cp = _CODE_PX.get(_st_cfg.get("font_size_code", "small"), 13)
    return f"""<style>
.stApp                                          {{ background-color: {c['bg']}; }}
[data-testid="stSidebar"]                       {{ background-color: {c['sidebar_bg']}; }}
.stApp, [data-testid="stSidebar"]               {{ color: {c['text']}; }}
.stMarkdown p, .stMarkdown li, .stMarkdown span {{ font-size: {fp}px; }}
.stTextArea textarea, .stTextInput input        {{ font-size: {fp}px; }}
code, pre, [data-testid="stCode"] *             {{ font-size: {cp}px !important; }}
</style>"""


# ------------------------------------------------------------------
# Применяем CSS и инициализируем кэш
# ------------------------------------------------------------------

st.markdown(_build_css(), unsafe_allow_html=True)


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
# Константы из конфига
# ------------------------------------------------------------------

_REFRESH_SIDEBAR   = _st_cfg.get("refresh_sidebar", "1s")
_REFRESH_FEED      = _st_cfg.get("refresh_feed", "0.5s")
_FEED_HEIGHT_FULL  = _st_cfg.get("feed_height_full", 720)
_FEED_HEIGHT_WITH_Q = _st_cfg.get("feed_height_with_question", 420)
_ANSWER_HEIGHT     = _st_cfg.get("answer_height", 88)


# ------------------------------------------------------------------
# Session state
# ------------------------------------------------------------------

if "bridge" not in st.session_state:
    st.session_state.bridge = AgentBridge()


# ------------------------------------------------------------------
# Сайдбар — управление (в фрагменте, чтобы кнопки отражали живое состояние)
# ------------------------------------------------------------------

with st.sidebar:
    st.title(_st_cfg.get("page_title", "Агентский движок"))

    bridge = st.session_state.bridge
    st.selectbox("Агент", bridge.available_agents(),
                 disabled=bridge.is_running, key="ui_agent_name")
    st.text_area("Стартовое сообщение",
                 placeholder="Опишите задачу для агента...",
                 disabled=bridge.is_running, key="ui_start_message")

    @st.fragment(run_every=_REFRESH_SIDEBAR)
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
            disabled=b.is_running,
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
# Основная область — лента сверху, вопрос/ответ снизу
# ------------------------------------------------------------------

st.header("Лента событий")


@st.fragment(run_every=_REFRESH_FEED)
def agent_panel() -> None:
    b: AgentBridge = st.session_state.bridge
    pq = b.pending_question

    feed_height = _FEED_HEIGHT_WITH_Q if pq is not None else _FEED_HEIGHT_FULL
    events = b.events
    with st.container(height=feed_height, border=True):
        if not events:
            st.caption("Событий пока нет. Запустите агента.")
        else:
            for event in events:
                _render_event(event)

    if pq is not None:
        with st.form("answer_form", clear_on_submit=True):
            st.warning(f"**Вопрос агента:** {pq}")
            answer = st.text_area("Ваш ответ", key="answer_input", height=_ANSWER_HEIGHT)
            submitted = st.form_submit_button("Отправить", type="primary")
        if submitted and answer.strip():
            b.send_answer(answer.strip())


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

    elif etype == "print":
        with st.chat_message("tool", avatar="🛠️"):
            text = event.get("text", "")
            st.markdown(text.replace("\n", "  \n"))

    elif etype == "warning":
        st.warning(event.get("message", ""))

    elif etype == "error":
        st.error(f"**Ошибка:** {event.get('message', '')}")

    elif etype == "image":
        path = event.get("path", "")
        caption = event.get("caption") or None
        if Path(path).exists():
            st.image(path, caption=caption)
        else:
            st.warning(f"Изображение не найдено: {path}")

    elif etype == "stopped":
        st.warning(event.get("message", "Остановлено"))

    elif etype == "llm_error":
        st.error(f"**Ошибка LLM:** {event.get('error', '')}")


agent_panel()
