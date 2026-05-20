"""EventGuard Command Center: Streamlit dashboard with high-volume metrics."""

import logging
import os
from datetime import datetime

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


# -- Authentication -------------------------------------------------------

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

    st.title("Login \u2014 EventGuard Command Center")
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


# -- Dashboard ------------------------------------------------------------

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


def _compute_detection_rate(df: pd.DataFrame) -> float:
    """Return faces-per-minute based on the earliest and latest log entries."""
    if df.empty or "Time" not in df.columns:
        return 0.0
    try:
        times = pd.to_datetime(df["Time"], format="%H:%M:%S", errors="coerce").dropna()
        if len(times) < 2:
            return 0.0
        span_minutes = (times.max() - times.min()).total_seconds() / 60.0
        if span_minutes <= 0:
            return 0.0
        return len(times) / span_minutes
    except Exception:
        return 0.0


def render_dashboard():
    """Render the main dashboard view."""
    init_db()

    st.title("EventGuard Command Center")

    total, inside, df = get_metrics()

    unique_faces = df["Name"].nunique() if not df.empty else 0
    detection_rate = _compute_detection_rate(df)

    m1, m2, m3, m4, m5 = st.columns(5)
    m1.metric("Total Guest List", total)

    percentage = f"{inside / total:.1%}" if total > 0 else "0%"
    m2.metric("Checked In", inside, delta=percentage)
    m3.metric("Remaining", max(total - inside, 0))
    m4.metric("Unique Faces", unique_faces)
    m5.metric("Rate (faces/min)", f"{detection_rate:.1f}")

    st.divider()
    st.subheader("Live Access Feed")

    col_filter, col_search = st.columns([1, 2])
    with col_filter:
        status_options = ["All"]
        if not df.empty and "Status" in df.columns:
            status_options += sorted(df["Status"].unique().tolist())
        selected_status = st.selectbox("Filter by status", status_options)

    with col_search:
        search_query = st.text_input("Search by name", "")

    filtered_df = df.copy()
    if selected_status != "All" and not filtered_df.empty:
        filtered_df = filtered_df[filtered_df["Status"] == selected_status]
    if search_query and not filtered_df.empty:
        filtered_df = filtered_df[
            filtered_df["Name"].str.contains(search_query, case=False, na=False)
        ]

    st.dataframe(
        filtered_df,
        use_container_width=True,
        hide_index=True,
    )

    st.caption(f"Showing {len(filtered_df)} of {len(df)} entries")

    if st.button("Refresh"):
        st.rerun()


# -- Entry point ----------------------------------------------------------

if check_auth():
    render_dashboard()
