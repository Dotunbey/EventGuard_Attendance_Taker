"""EventGuard Command Center: Streamlit dashboard with authentication."""

import logging
import os

import pandas as pd
import streamlit as st

from src.config import (
    DASHBOARD_PASSWORD,
    DASHBOARD_USERNAME,
    setup_logging,
)
from src.database import (
    get_access_log,
    get_encoding_count,
    init_db,
)

setup_logging()
logger = logging.getLogger(__name__)

st.set_page_config(page_title="EventGuard Ops", layout="wide")


# ── Authentication ───────────────────────────────────────────────

def _get_credentials() -> tuple:
    """Resolve dashboard credentials from st.secrets or env vars."""
    username = ""
    password = ""
    try:
        username = st.secrets.get("dashboard_username", "")
        password = st.secrets.get("dashboard_password", "")
    except Exception:
        pass
    if not username:
        username = DASHBOARD_USERNAME
    if not password:
        password = DASHBOARD_PASSWORD
    return username, password


def check_auth() -> bool:
    """Return True if the user is authenticated or auth is not configured."""
    username, password = _get_credentials()

    if not username and not password:
        return True

    if st.session_state.get("authenticated"):
        return True

    st.title("Login — EventGuard Command Center")
    with st.form("login_form"):
        input_user = st.text_input("Username")
        input_pass = st.text_input("Password", type="password")
        submitted = st.form_submit_button("Log In")

    if submitted:
        if input_user == username and input_pass == password:
            st.session_state["authenticated"] = True
            st.rerun()
        else:
            st.error("Invalid credentials")

    return False


# ── Dashboard ────────────────────────────────────────────────────

def get_metrics():
    """Fetch metrics from SQLite database."""
    try:
        total_invited = get_encoding_count()
    except Exception:
        logger.exception("Failed to get encoding count")
        total_invited = 0

    try:
        log_rows = get_access_log()
        df = pd.DataFrame(log_rows) if log_rows else pd.DataFrame(
            columns=["Name", "Time", "Status"]
        )
    except Exception:
        logger.exception("Failed to get access log")
        df = pd.DataFrame(columns=["Name", "Time", "Status"])

    return total_invited, len(df), df


def render_dashboard():
    """Render the main dashboard view."""
    init_db()

    st.title("EventGuard Command Center")

    total, inside, df = get_metrics()

    m1, m2, m3 = st.columns(3)
    m1.metric("Total Guest List", total)

    percentage = f"{inside / total:.1%}" if total > 0 else "0%"
    m2.metric("Checked In", inside, delta=percentage)
    m3.metric("Remaining", max(total - inside, 0))

    st.divider()
    st.subheader("Live Access Feed")

    st.dataframe(
        df,
        use_container_width=True,
        hide_index=True,
    )

    if st.button("Refresh"):
        st.rerun()


# ── Entry point ──────────────────────────────────────────────────

if check_auth():
    render_dashboard()
