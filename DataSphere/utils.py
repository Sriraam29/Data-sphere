import streamlit as st
import json
import os
import tempfile

# File-based persistence — survives Streamlit reruns within the same server session.
# Uses a temp file keyed to the Streamlit session so multiple users don't collide.
_PERSIST_DIR = os.path.join(tempfile.gettempdir(), "datasphere_sessions")

def _get_persist_path() -> str:
    """Return a per-session file path for persisted data."""
    os.makedirs(_PERSIST_DIR, exist_ok=True)
    # Use a fixed filename per process; fine for single-user / dev deployments.
    return os.path.join(_PERSIST_DIR, "session_data.json")


def initialize_session_state():
    """Initialize ALL session state variables if they don't already exist.
    This is the single source of truth — no other module should set defaults.
    """
    defaults = {
        "db_connections": {},
        "connected_db": None,
        "current_connection": None,
        "db_schema": None,
        "query_results": None,
        "current_query": "",
        "natural_language_query": "",
        "query_history": [],
        "available_databases": [],
    }
    for key, default in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = default


def save_session_state():
    """Persist connection metadata and query history to a temp file.
    Passwords are NOT saved — they live only in session memory.
    """
    # Strip passwords from connection params before saving
    safe_connections = {}
    for name, details in st.session_state.db_connections.items():
        safe_params = {k: v for k, v in details.get("connection_params", {}).items() if k != "password"}
        safe_connections[name] = {
            k: v for k, v in details.items() if k not in ("connection_string",)
        }
        safe_connections[name]["connection_params"] = safe_params

    save_data = {
        "db_connections": safe_connections,
        "query_history": st.session_state.query_history,
    }

    try:
        with open(_get_persist_path(), "w") as f:
            json.dump(save_data, f, indent=2)
    except Exception as e:
        st.warning(f"Could not save session data: {e}")


def load_session_state():
    """Load persisted connection metadata and query history from disk.
    Called once on app startup — skipped silently if no file exists yet.
    """
    path = _get_persist_path()
    if not os.path.exists(path):
        return

    try:
        with open(path) as f:
            saved_data = json.load(f)

        for key in ("db_connections", "query_history"):
            if key in saved_data:
                st.session_state[key] = saved_data[key]
    except Exception as e:
        st.warning(f"Could not load saved session data: {e}")