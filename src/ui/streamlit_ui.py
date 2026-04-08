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
from src.agents import build_agent                          # noqa: E402


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
if "cp_selected_session" not in st.session_state:
    st.session_state.cp_selected_session = None
if "cp_selected_checkpoint" not in st.session_state:
    st.session_state.cp_selected_checkpoint = None
if "cp_edit_mode" not in st.session_state:
    st.session_state.cp_edit_mode = False
if "cp_rename_mode" not in st.session_state:
    st.session_state.cp_rename_mode = False


# ------------------------------------------------------------------
# SessionManager helper (кэшируется)
# ------------------------------------------------------------------

def _get_session_manager():
    """Вернуть SessionManager из конфига или None."""
    try:
        from src.agent_engine.session_manager import SessionManager
        cfg = _load_config()
        sessions_cfg = cfg.get("sessions", {})
        if not sessions_cfg.get("enabled", False):
            return None
        db_path = _PROJECT_ROOT / sessions_cfg.get("db_path", "sessions/checkpoints.db")
        reg_path = _PROJECT_ROOT / sessions_cfg.get("registry_path", "sessions/sessions.json")
        return SessionManager(db_path=db_path, registry_path=reg_path)
    except Exception:
        return None


@st.cache_resource
def _get_cached_session_manager():
    return _get_session_manager()


def _start_from_checkpoint(agent, agent_name: str, session_id: str,
                            checkpoint_id: str, edits: dict | None) -> None:
    """Тегировать чекпоинт, форкнуть и запустить в bridge."""
    import uuid as _uuid
    bridge: AgentBridge = st.session_state.bridge
    if bridge.is_running:
        st.warning("Агент уже запущен. Остановите текущий запуск.")
        return
    try:
        temp_name = f"_ui_fork_{_uuid.uuid4().hex[:8]}"
        agent.tag_checkpoint(session_id, temp_name, checkpoint_id=checkpoint_id)
        new_sid = agent.fork(temp_name, edits=edits)
        agent._sm.delete_named_checkpoint(temp_name)

        cfg = _load_config()
        recursion_limit = cfg.get("agent", {}).get("recursion_limit", 100)
        llm = _init_llm()
        # Запускаем с пустым сообщением — граф продолжит из форкнутого checkpoint
        bridge.start(agent_name, "", llm, recursion_limit, session_id=new_sid)
        st.session_state.cp_edit_mode = False
        st.rerun(scope="app")
    except Exception as e:
        st.error(f"Ошибка запуска из чекпоинта: {e}")


# ------------------------------------------------------------------
# Панель сессий и чекпоинтов (вызывается из sidebar)
# ------------------------------------------------------------------

def _render_sessions_panel() -> None:
    sm = _get_cached_session_manager()
    if sm is None:
        return

    st.divider()
    with st.expander("Сессии и чекпоинты", expanded=False):
        agent_name: str = st.session_state.get("ui_agent_name", "")
        if not agent_name:
            st.caption("Выберите агента выше")
            return

        sessions = sm.list_sessions(agent_name)
        if not sessions:
            st.caption("Нет сохранённых сессий")
            bridge: AgentBridge = st.session_state.bridge
            if bridge.last_session_id:
                st.caption(f"Последняя: `{bridge.last_session_id[:8]}...`")
            return

        # --- Список сессий ---
        session_labels = []
        for s in sessions:
            label = s.get("description") or s["session_id"][:8]
            ts = s.get("updated_at", "")[:16].replace("T", " ")
            named_count = len(sm.list_named_checkpoints(s["session_id"]))
            star = " ★" if named_count > 0 else ""
            session_labels.append(f"{ts} | {label}{star}")

        sel_idx = st.selectbox(
            "Сессия",
            range(len(sessions)),
            format_func=lambda i: session_labels[i],
            key="cp_session_idx",
        )
        selected_session = sessions[sel_idx]
        sid = selected_session["session_id"]

        st.caption(f"ID: `{sid[:12]}...` | последнее: {selected_session.get('last_state') or '?'}")

        if st.button("🗑 Удалить сессию", key="btn_del_session", use_container_width=True):
            sm.delete_session(sid)
            st.rerun()

        st.markdown("**Чекпоинты:**")

        # Загружаем чекпоинты через граф агента
        try:
            llm = _init_llm()
            agent = build_agent(agent_name, llm)
            checkpoints = agent.list_checkpoints(sid)
        except Exception as e:
            st.warning(f"Не удалось загрузить чекпоинты: {e}")
            return

        if not checkpoints:
            st.caption("Нет чекпоинтов")
            return

        cp_labels = []
        for cp in checkpoints:
            ts = cp.get("timestamp", "")[:19].replace("T", " ")
            name_tag = f" ★ {cp['name']}" if cp.get("is_named") else ""
            cp_labels.append(f"{cp['state_name']}{name_tag}  |  {ts}")

        cp_idx = st.selectbox(
            "Чекпоинт",
            range(len(checkpoints)),
            format_func=lambda i: cp_labels[i],
            key="cp_checkpoint_idx",
        )
        selected_cp = checkpoints[cp_idx]
        cp_id = selected_cp["checkpoint_id"]
        st.caption(f"`{cp_id[:12]}...`")

        # --- Кнопки действий ---
        col1, col2 = st.columns(2)
        col3, col4 = st.columns(2)

        if col1.button("👁 Просмотр", key="btn_cp_view", use_container_width=True):
            try:
                state_vals = agent.get_checkpoint_state_by_id(sid, cp_id)
                bridge: AgentBridge = st.session_state.bridge
                import json as _json
                bridge._handle_event({
                    "type": "print",
                    "text": (
                        f"**Чекпоинт: {selected_cp['state_name']}**  "
                        f"`{cp_id[:8]}...`  {selected_cp.get('timestamp', '')[:19]}\n\n"
                        f"**memory:** ```json\n{_json.dumps(state_vals.get('memory', {}), ensure_ascii=False, indent=2)}\n```\n\n"
                        f"**summary:** {state_vals.get('summary', '') or '—'}"
                    ),
                })
                st.rerun(scope="app")
            except Exception as e:
                st.error(f"Ошибка: {e}")

        if col2.button("▶ Старт", key="btn_cp_start", use_container_width=True):
            _start_from_checkpoint(agent, agent_name, sid, cp_id, edits=None)

        rename_label = f"✏ [{selected_cp['name']}]" if selected_cp.get("is_named") else "✏ Назвать"
        if col3.button(rename_label, key="btn_cp_rename", use_container_width=True):
            st.session_state.cp_rename_mode = not st.session_state.cp_rename_mode
            st.session_state.cp_edit_mode = False

        if col4.button("🔧 Редактировать", key="btn_cp_edit", use_container_width=True):
            st.session_state.cp_edit_mode = not st.session_state.cp_edit_mode
            st.session_state.cp_rename_mode = False

        # --- Панель переименования / тегирования ---
        if st.session_state.cp_rename_mode:
            with st.form("rename_form"):
                default_name = selected_cp.get("name", "")
                new_name = st.text_input("Имя чекпоинта", value=default_name,
                                         placeholder="baseline_v1")
                note = st.text_input("Заметка (опционально)")
                submitted = st.form_submit_button("Сохранить")
            if submitted and new_name.strip():
                try:
                    if selected_cp.get("is_named") and default_name:
                        if new_name.strip() != default_name:
                            sm.rename_checkpoint(default_name, new_name.strip())
                    else:
                        sm.save_named_checkpoint(sid, new_name.strip(), cp_id, note=note.strip())
                    st.session_state.cp_rename_mode = False
                    st.success(f"Сохранено: '{new_name.strip()}'")
                    st.rerun()
                except Exception as e:
                    st.error(f"Ошибка: {e}")

        if selected_cp.get("is_named"):
            if st.button(f"🗑 Убрать имя '{selected_cp['name']}'", key="btn_del_named"):
                sm.delete_named_checkpoint(selected_cp["name"])
                st.rerun()

        # --- Панель редактирования ---
        if st.session_state.cp_edit_mode:
            st.markdown("**Изменить состояние перед стартом:**")
            try:
                import json as _json
                state_vals = agent.get_checkpoint_state_by_id(sid, cp_id)
                memory_str = st.text_area(
                    "memory (JSON)",
                    value=_json.dumps(state_vals.get("memory", {}),
                                      ensure_ascii=False, indent=2),
                    height=160,
                    key="cp_edit_memory",
                )
                summary_str = st.text_area(
                    "summary",
                    value=state_vals.get("summary", ""),
                    height=80,
                    key="cp_edit_summary",
                )
                if st.button("▶ Стартовать с правками", key="btn_cp_start_edited",
                             use_container_width=True, type="primary"):
                    try:
                        edits: dict = {}
                        if memory_str.strip():
                            edits["memory"] = _json.loads(memory_str)
                        orig_summary = state_vals.get("summary", "")
                        if summary_str.strip() != orig_summary:
                            edits["summary"] = summary_str.strip()
                        _start_from_checkpoint(
                            agent, agent_name, sid, cp_id,
                            edits=edits if edits else None,
                        )
                    except _json.JSONDecodeError as e:
                        st.error(f"Некорректный JSON: {e}")
                    except Exception as e:
                        st.error(f"Ошибка: {e}")
            except Exception as e:
                st.error(f"Не удалось загрузить состояние: {e}")


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
    # Панель сессий и чекпоинтов
    # ------------------------------------------------------------------
    _render_sessions_panel()


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
