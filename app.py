"""Streamlit entry point for the OCR Accuracy Comparison Demo."""

from __future__ import annotations

import asyncio
import json
import mimetypes
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
from uuid import uuid4

import pandas as pd
import streamlit as st

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

from src.auth import (  # noqa: E402
    approve_passcode_request,
    authenticate_user,
    delete_user_activity,
    deny_passcode_request,
    ensure_storage,
    list_users,
    load_admin_audit,
    load_login_activity,
    load_passcode_requests,
    load_user_activity,
    log_admin_action,
    log_login_attempt,
    log_user_activity,
    login_summary,
    soft_delete_user,
    submit_passcode_request,
    update_user,
)
from src.config import load_config  # noqa: E402
from src.metrics.llm_judge import judge as judge_call  # noqa: E402
from src.orchestrator import run_all  # noqa: E402
from src.pipelines.base import DocumentInput, PipelineResult  # noqa: E402
from src.pipelines.doc_intelligence import DocIntelligencePipeline  # noqa: E402
from src.pipelines.hybrid import HybridDIPipeline  # noqa: E402
from src.pipelines.llm_vision import LLMVisionPipeline  # noqa: E402
from src.preprocess import preprocess  # noqa: E402
from src.storage import delete_run, list_runs, load_run, save_run  # noqa: E402
from src.ui.results_view import render_results  # noqa: E402
from src.ui.sidebar import render_sidebar  # noqa: E402


st.set_page_config(
    page_title="Smart OCR for Serious Documents",
    page_icon="📄",
    layout="wide",
)

st.markdown(
    """<style>
    :root {
      --bg: #eaf1fb;
      --surface: #ffffff;
      --ink: #1e2948;
      --muted: #60739b;
      --primary: #214aa7;
      --primary-2: #173780;
      --border: #d6e1f4;
      --danger: #b24020;
    }
    html, body, [class*="stApp"] {
      background: var(--bg) !important;
      color: var(--ink) !important;
    }
    .block-container { max-width: 1500px; padding-top: 3.5rem !important; padding-bottom: 7rem !important; }
    .stButton > button, .stDownloadButton > button {
      background: var(--primary) !important;
      color: #fff !important;
      border: 1px solid var(--primary) !important;
      border-radius: 999px !important;
      font-weight: 600 !important;
    }
    .stButton > button:hover, .stDownloadButton > button:hover {
      background: var(--primary-2) !important;
      border-color: var(--primary-2) !important;
      color: #fff !important;
    }
    .stButton > button p, .stButton > button span,
    .stDownloadButton > button p, .stDownloadButton > button span {
      color: #fff !important;
    }
    .app-footer {
      position: fixed;
      left: 22rem;
      right: 14px;
      bottom: 10px;
      padding: 10px 16px;
      border: 1px solid var(--border);
      border-radius: 14px;
      background: rgba(255,255,255,0.92);
      color: var(--ink);
      font-size: 0.85rem;
      line-height: 1.35;
      text-align: center;
      white-space: normal;
      overflow-wrap: anywhere;
      z-index: 1000;
    }
    /* When the sidebar is collapsed, reclaim the full width. */
    body:has(section[data-testid="stSidebar"][aria-expanded="false"]) .app-footer {
      left: 14px;
    }
    .app-footer a {
      color: var(--primary);
      font-weight: 700;
      text-decoration: none;
    }
    .auth-shell {
      padding: 0;
      margin-top: -0.25rem;
    }
    .app-topbar {
      margin: 0 0 0.85rem;
      padding: 0 0 0.4rem;
    }
    .app-topbar-title {
      font-size: 1.55rem;
      line-height: 1.3;
      font-weight: 800;
      color: var(--ink);
    }
    .app-topbar-sub {
      font-size: 0.95rem;
      color: var(--muted);
      margin-top: 0.15rem;
    }
    .auth-hero-card {
      background: transparent;
      border: none;
      padding: 0.75rem 0 0.5rem;
      box-shadow: none;
      text-align: center;
    }
    .auth-form-card {
      background: linear-gradient(180deg, #ffffff 0%, #f6f9ff 100%);
      border: 1px solid var(--border);
      border-radius: 18px;
      padding: 0.8rem 0.9rem;
      box-shadow: 0 8px 18px rgba(33, 74, 167, 0.08);
    }
    .auth-hero-badge {
      display: inline-flex;
      align-items: center;
      gap: 0.4rem;
      padding: 0.35rem 0.65rem;
      border-radius: 999px;
      background: #edf3ff;
      color: var(--primary);
      font-weight: 700;
      font-size: 0.92rem;
      letter-spacing: 0.02em;
      margin-bottom: 0.6rem;
    }
    .auth-hero-title {
      font-size: 1.9rem;
      line-height: 1.35;
      padding-top: 0.35rem;
      margin-bottom: 0.35rem;
      color: var(--ink);
      font-weight: 800;
    }
    .auth-hero-subtitle {
      color: var(--muted);
      font-size: 1rem;
      margin: 0 auto;
      max-width: 38rem;
    }
    .auth-form-card h3, .auth-form-card h4 {
      color: var(--ink);
      margin-bottom: 0.1rem;
    }
    .auth-note {
      border-left: 3px solid var(--primary);
      padding: 0.45rem 0.7rem;
      border-radius: 10px;
      background: #f4f7ff;
      color: var(--ink);
      font-size: 0.95rem;
    }
    </style>""",
    unsafe_allow_html=True,
)


def _init_session_state() -> None:
    defaults = {
        "authenticated": False,
        "auth_username": "",
        "auth_role": "",
        "request_form_open": False,
        "active_top_nav": "Compare",
        "workspace_mode": "Workspace",
        "last_results": None,
        "last_gt_text": None,
        "last_gt_json": None,
        "last_judge_scores": None,
        "last_uploaded_name": "",
        "doctalk_history": [],
    }
    for key, value in defaults.items():
        st.session_state.setdefault(key, value)


def _logout() -> None:
    for key in [
        "authenticated",
        "auth_username",
        "auth_role",
        "request_form_open",
        "workspace_mode",
        "active_top_nav",
        "last_results",
        "last_gt_text",
        "last_gt_json",
        "last_judge_scores",
        "last_uploaded_name",
        "doctalk_history",
    ]:
        st.session_state.pop(key, None)
    _init_session_state()


def _render_app_header() -> None:
    st.markdown(
        """
        <div class='app-topbar'>
          <div class='app-topbar-title'>📄 Smart OCR for Serious Documents</div>
          <div class='app-topbar-sub'>Compare OCR pipelines, inspect confidence and accuracy, and review extractions.</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def _render_top_nav() -> str:
    options = ["Compare", "Determinism", "Insights", "History"]
    current = st.session_state.get("active_top_nav", "Compare")
    segmented_control = getattr(st, "segmented_control", None)
    if callable(segmented_control):
        choice = segmented_control(
            "Navigation",
            options=options,
            default=current if current in options else options[0],
            key="active_top_nav",
            label_visibility="collapsed",
        )
    else:
        choice = st.radio(
            "Navigation",
            options,
            horizontal=True,
            index=options.index(current) if current in options else 0,
            key="active_top_nav",
            label_visibility="collapsed",
        )
    return choice or st.session_state.get("active_top_nav", "Compare")


@st.dialog("Request passcode")
def _request_passcode_dialog() -> None:
    st.caption("Send a short access request for admin review.")
    request_name = st.text_input("Name", key="request_name")
    request_contact = st.text_input("Contact or email", key="request_contact")
    request_reason = st.text_area(
        "Why do you need access?", key="request_reason", height=120
    )
    submit_col, cancel_col = st.columns(2)
    submit_clicked = submit_col.button(
        "Submit request",
        type="primary",
        key="request_passcode_btn",
        use_container_width=True,
    )
    cancel_clicked = cancel_col.button(
        "Cancel", key="cancel_request_form_btn", use_container_width=True
    )

    if submit_clicked:
        if not request_name.strip() or not request_contact.strip():
            st.warning("Name and contact are required.")
        else:
            submit_passcode_request(request_name, request_contact, request_reason)
            st.session_state["request_submitted"] = True
            st.rerun()
    if cancel_clicked:
        st.rerun()


def _render_login_page() -> None:
    st.markdown(
        """
        <style>
        /* ===== Login brand tone: ICICI orange + Axis burgundy ===== */
        :root {
          --brand-burgundy: #8C0A3E;
          --brand-magenta: #B02A5B;
          --brand-orange: #F37021;
          --brand-gradient: linear-gradient(135deg, #8C0A3E 0%, #B02A5B 48%, #F37021 100%);
        }
        /* Soft branded backdrop for the login screen only */
        body:has(.login-brand) [data-testid="stAppViewContainer"] {
          background:
            radial-gradient(1100px 540px at 12% -8%, rgba(140, 10, 62, 0.12), transparent 60%),
            radial-gradient(1000px 520px at 92% 6%, rgba(243, 112, 33, 0.13), transparent 58%),
            #fbf5f3 !important;
        }
        .login-brand .auth-hero-card {
          position: relative;
          background: var(--brand-gradient);
          border: none;
          border-radius: 22px;
          padding: 1.6rem 1.5rem 1.7rem;
          text-align: center;
          color: #fff;
          box-shadow: 0 18px 40px rgba(140, 10, 62, 0.28);
          overflow: hidden;
        }
        .login-brand .auth-hero-card::after {
          content: "";
          position: absolute;
          inset: 0;
          background: radial-gradient(420px 200px at 85% 0%, rgba(255,255,255,0.18), transparent 60%);
          pointer-events: none;
        }
        .login-brand .auth-hero-badge {
          display: inline-flex;
          align-items: center;
          gap: 0.4rem;
          padding: 0.32rem 0.8rem;
          border-radius: 999px;
          background: rgba(255, 255, 255, 0.18);
          color: #fff;
          font-weight: 700;
          font-size: 0.85rem;
          letter-spacing: 0.04em;
          text-transform: uppercase;
          margin-bottom: 0.8rem;
          backdrop-filter: blur(2px);
        }
        .login-brand .auth-hero-title {
          font-size: 2rem;
          line-height: 1.25;
          font-weight: 800;
          color: #fff;
          margin: 0 0 0.5rem;
          text-shadow: 0 2px 10px rgba(0,0,0,0.12);
        }
        .login-brand .auth-hero-subtitle {
          color: rgba(255, 255, 255, 0.92);
          font-size: 1rem;
          margin: 0 auto;
          max-width: 36rem;
        }
        /* Card holding the form */
        body:has(.login-brand-anchor) [data-testid="stVerticalBlockBorderWrapper"] {
          border: 1px solid #f0d8de !important;
          border-radius: 18px !important;
          background: linear-gradient(180deg, #ffffff 0%, #fff7f4 100%) !important;
          box-shadow: 0 12px 28px rgba(140, 10, 62, 0.10) !important;
        }
        body:has(.login-brand-anchor) [data-testid="stVerticalBlockBorderWrapper"] h4 {
          color: var(--brand-burgundy) !important;
        }
        /* Inputs pick up the burgundy focus ring */
        body:has(.login-brand-anchor) [data-testid="stTextInput"] input:focus {
          border-color: var(--brand-magenta) !important;
          box-shadow: 0 0 0 2px rgba(176, 42, 91, 0.25) !important;
        }
        /* Primary "Sign in" -> burgundy→orange gradient */
        body:has(.login-brand-anchor) .stButton button[kind="primary"] {
          background: var(--brand-gradient) !important;
          border: none !important;
          box-shadow: 0 8px 18px rgba(140, 10, 62, 0.30) !important;
        }
        body:has(.login-brand-anchor) .stButton button[kind="primary"]:hover {
          filter: brightness(1.05);
          box-shadow: 0 10px 22px rgba(243, 112, 33, 0.34) !important;
        }
        /* Secondary "Request passcode" -> outlined burgundy */
        body:has(.login-brand-anchor) .stButton button[kind="secondary"] {
          background: #ffffff !important;
          color: var(--brand-burgundy) !important;
          border: 1.5px solid var(--brand-burgundy) !important;
        }
        body:has(.login-brand-anchor) .stButton button[kind="secondary"] p,
        body:has(.login-brand-anchor) .stButton button[kind="secondary"] span {
          color: var(--brand-burgundy) !important;
        }
        body:has(.login-brand-anchor) .stButton button[kind="secondary"]:hover {
          background: #fbeef2 !important;
          border-color: var(--brand-magenta) !important;
        }
        </style>
        <div class='login-brand-anchor'></div>
        """,
        unsafe_allow_html=True,
    )

    _, center_col, _ = st.columns([1, 1.6, 1])

    with center_col:
        st.markdown(
            """
            <div class='login-brand'>
              <div class='auth-hero-card'>
                <div class='auth-hero-badge'>📄 OCR Accuracy Workspace</div>
                <div class='auth-hero-title'>Smart OCR for Serious Documents</div>
                <p class='auth-hero-subtitle'>Sign in to compare OCR pipelines, inspect confidence and accuracy, and continue your review.</p>
              </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

        with st.container(border=True):
            st.markdown("#### Welcome back")
            st.caption("Use your username and passcode to enter the workspace.")

            if st.session_state.pop("request_submitted", False):
                st.success("Request submitted for admin review.")

            username = st.text_input("Username", key="login_username")
            passcode = st.text_input("Passcode", type="password", key="login_passcode")

            sign_in_col, request_col = st.columns(2, gap="small")
            with sign_in_col:
                sign_in_clicked = st.button(
                    "Sign in", type="primary", key="login_btn", use_container_width=True
                )
            with request_col:
                request_clicked = st.button(
                    "Request passcode",
                    key="open_request_form_btn",
                    use_container_width=True,
                )

            if sign_in_clicked:
                user = authenticate_user(username, passcode)
                if user is not None:
                    st.session_state["authenticated"] = True
                    st.session_state["auth_username"] = str(user.get("username") or "")
                    st.session_state["auth_role"] = str(user.get("role") or "user")
                    log_login_attempt(
                        username, True, role=st.session_state["auth_role"]
                    )
                    st.rerun()
                log_login_attempt(username, False, reason="invalid_credentials")
                st.error("Invalid username or passcode.")

            if request_clicked:
                _request_passcode_dialog()


def _events_to_frame(events: list[dict[str, Any]]) -> pd.DataFrame:
    if not events:
        return pd.DataFrame()
    return pd.DataFrame(events)


def _render_admin_page() -> None:
    st.subheader("Admin page")
    summary = login_summary()
    requests = load_passcode_requests()
    users = list_users(include_inactive=True)
    activity_events = load_user_activity()
    login_events = load_login_activity()

    pending_count = sum(
        1 for row in requests if str(row.get("status", "")).lower() == "pending"
    )
    active_users = sum(1 for user in users if bool(user.get("active", True)))
    run_count = sum(
        1 for event in activity_events if event.get("action") == "run_completed"
    )
    failed_logins = sum(1 for event in login_events if not event.get("success"))

    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Active users", active_users)
    m2.metric("Pending requests", pending_count)
    m3.metric("Document runs", run_count)
    m4.metric("Failed logins", failed_logins)

    req_tab, users_tab, activity_tab, login_tab, audit_tab = st.tabs(
        ["Requests", "Users", "Activity logs", "Login logs", "Admin audit"]
    )

    with req_tab:
        st.markdown("#### Passcode requests")
        if requests:
            st.dataframe(requests, use_container_width=True, hide_index=True)
            pending_indices = [
                i
                for i, row in enumerate(requests)
                if str(row.get("status", "")).lower() == "pending"
            ]
            if pending_indices:
                st.markdown("##### Approve or deny pending requests")
                for request_index in pending_indices:
                    row = requests[request_index]
                    requester_name = str(row.get("name") or "unknown")
                    requester_contact = str(row.get("contact") or "")
                    suggested = "".join(
                        ch.lower() for ch in requester_name if ch.isalnum()
                    )
                    st.markdown(
                        f"**Request #{request_index + 1}** | {requester_name} ({requester_contact})"
                    )
                    c1, c2, c3 = st.columns([2, 1, 2])
                    username_input = c1.text_input(
                        "Username",
                        value=suggested,
                        key=f"approve_username_{request_index}",
                    )
                    role_input = c2.selectbox(
                        "Role",
                        ["user", "admin"],
                        index=0,
                        key=f"approve_role_{request_index}",
                    )
                    denial_reason = c3.text_input(
                        "Denial reason", key=f"deny_reason_{request_index}"
                    )

                    p1, p2 = st.columns(2)
                    passcode_input = p1.text_input(
                        "Passcode",
                        type="password",
                        key=f"approve_passcode_{request_index}",
                    )
                    approve_clicked = p2.button(
                        "Approve", key=f"approve_btn_{request_index}"
                    )
                    deny_clicked = p2.button("Deny", key=f"deny_btn_{request_index}")

                    if approve_clicked:
                        if not passcode_input.strip():
                            st.error("Passcode is required.")
                        else:
                            try:
                                created = approve_passcode_request(
                                    request_index,
                                    st.session_state.get("auth_username", "admin"),
                                    passcode_input,
                                    username_override=username_input,
                                    role=role_input,
                                )
                                log_admin_action(
                                    st.session_state.get("auth_username", "admin"),
                                    "approve_user_request",
                                    target_username=created["username"],
                                    details={
                                        "request_index": request_index,
                                        "role": role_input,
                                    },
                                )
                                st.success(f"User created: {created['username']}")
                                st.rerun()
                            except (ValueError, IndexError) as exc:
                                st.error(str(exc))

                    if deny_clicked:
                        try:
                            deny_passcode_request(
                                request_index,
                                st.session_state.get("auth_username", "admin"),
                                reason=denial_reason,
                            )
                            log_admin_action(
                                st.session_state.get("auth_username", "admin"),
                                "deny_user_request",
                                details={
                                    "request_index": request_index,
                                    "reason": denial_reason,
                                },
                            )
                            st.warning("Request denied.")
                            st.rerun()
                        except (ValueError, IndexError) as exc:
                            st.error(str(exc))
                    st.divider()
            else:
                st.caption("No pending passcode requests.")
        else:
            st.info("No passcode requests yet.")

    with users_tab:
        st.markdown("#### Users")
        if not users:
            st.info("No users found.")
        else:
            display_users = [
                {
                    "username": user.get("username", ""),
                    "role": user.get("role", "user"),
                    "active": bool(user.get("active", True)),
                    "created_at": user.get("created_at", ""),
                    "updated_at": user.get("updated_at", ""),
                }
                for user in users
            ]
            st.dataframe(display_users, use_container_width=True, hide_index=True)

            st.markdown("##### Edit role / reset passcode / delete user")
            for user in users:
                username = str(user.get("username", ""))
                status = "active" if bool(user.get("active", True)) else "inactive"
                with st.expander(f"Manage {username} ({status})", expanded=False):
                    role_default = 0 if str(user.get("role", "user")) == "user" else 1
                    new_role = st.selectbox(
                        "Role",
                        ["user", "admin"],
                        index=role_default,
                        key=f"user_role_{username}",
                    )
                    reset_pass = st.text_input(
                        "Reset passcode", type="password", key=f"reset_pass_{username}"
                    )
                    col_save, col_delete = st.columns(2)
                    if col_save.button("Save changes", key=f"save_user_{username}"):
                        try:
                            update_user(
                                username,
                                role=new_role,
                                new_passcode=reset_pass if reset_pass.strip() else None,
                                updated_by=st.session_state.get(
                                    "auth_username", "admin"
                                ),
                            )
                            log_admin_action(
                                st.session_state.get("auth_username", "admin"),
                                "update_user",
                                target_username=username,
                                details={
                                    "role": new_role,
                                    "passcode_reset": bool(reset_pass.strip()),
                                },
                            )
                            st.success(f"Updated user: {username}")
                            st.rerun()
                        except ValueError as exc:
                            st.error(str(exc))

                    if bool(user.get("active", True)):
                        if col_delete.button(
                            "Delete user", key=f"delete_user_{username}"
                        ):
                            if username == st.session_state.get("auth_username", ""):
                                st.error("You cannot delete your own active account.")
                            else:
                                try:
                                    soft_delete_user(
                                        username,
                                        st.session_state.get("auth_username", "admin"),
                                    )
                                    st.warning(f"User deleted (soft): {username}")
                                    st.rerun()
                                except ValueError as exc:
                                    st.error(str(exc))
                    else:
                        if col_delete.button(
                            "Reactivate", key=f"reactivate_user_{username}"
                        ):
                            try:
                                update_user(
                                    username,
                                    active=True,
                                    updated_by=st.session_state.get(
                                        "auth_username", "admin"
                                    ),
                                )
                                st.success(f"User reactivated: {username}")
                                st.rerun()
                            except ValueError as exc:
                                st.error(str(exc))

    with activity_tab:
        st.markdown("#### User activity logs")
        if not activity_events:
            st.info("No user activity recorded yet.")
        else:
            indexed = [dict(event, _idx=i) for i, event in enumerate(activity_events)]
            df = _events_to_frame(indexed)
            df["timestamp"] = pd.to_datetime(
                df.get("timestamp"), errors="coerce", utc=True
            )

            usernames = sorted(
                [
                    x
                    for x in df["username"].fillna("").astype(str).unique().tolist()
                    if x
                ]
            )
            actions = sorted(
                [x for x in df["action"].fillna("").astype(str).unique().tolist() if x]
            )
            statuses = sorted(
                [x for x in df["status"].fillna("").astype(str).unique().tolist() if x]
            )

            f1, f2, f3, f4 = st.columns([1.2, 1.2, 1.2, 1.4])
            user_filter = f1.selectbox("User", ["All"] + usernames, index=0)
            action_filter = f2.selectbox("Action", ["All"] + actions, index=0)
            status_filter = f3.selectbox("Status", ["All"] + statuses, index=0)
            file_filter = f4.text_input("File contains", value="")

            d1, d2 = st.columns(2)
            ts_non_null = df["timestamp"].dropna()
            if ts_non_null.empty:
                min_date = pd.Timestamp.utcnow().date()
                max_date = min_date
            else:
                min_date = ts_non_null.dt.date.min()
                max_date = ts_non_null.dt.date.max()
            start_date = d1.date_input("From", value=min_date, key="activity_from")
            end_date = d2.date_input("To", value=max_date, key="activity_to")

            filtered = df.copy()
            if user_filter != "All":
                filtered = filtered[filtered["username"] == user_filter]
            if action_filter != "All":
                filtered = filtered[filtered["action"] == action_filter]
            if status_filter != "All":
                filtered = filtered[filtered["status"] == status_filter]
            if file_filter.strip():
                filtered = filtered[
                    filtered["filename"]
                    .fillna("")
                    .str.contains(file_filter.strip(), case=False)
                ]
            filtered = filtered[
                (filtered["timestamp"].dt.date >= start_date)
                & (filtered["timestamp"].dt.date <= end_date)
            ]
            filtered = filtered.sort_values("timestamp", ascending=False)

            st.dataframe(
                filtered.drop(columns=["_idx"]),
                use_container_width=True,
                hide_index=True,
            )

            run_df = filtered[filtered["action"] == "run_completed"]
            if not run_df.empty:
                per_user = run_df.groupby("username", as_index=False).agg(
                    runs=("action", "count"),
                    files=("filename", "nunique"),
                )
                st.markdown("##### User-wise run summary")
                st.dataframe(per_user, use_container_width=True, hide_index=True)

            c1, c2, c3 = st.columns(3)
            c1.download_button(
                "Download CSV",
                data=filtered.drop(columns=["_idx"]).to_csv(index=False),
                file_name="user_activity_logs.csv",
                mime="text/csv",
            )
            html_report = f"""
<html><head><title>User Activity Report</title></head><body>
<h2>User Activity Report</h2>
<p>Generated by: {st.session_state.get("auth_username", "admin")}</p>
<p>Generated at: {datetime.now(timezone.utc).isoformat()}</p>
<p>Rows: {len(filtered)}</p>
{filtered.drop(columns=["_idx"]).to_html(index=False)}
</body></html>
"""
            c2.download_button(
                "Download printable HTML",
                data=html_report,
                file_name="user_activity_report.html",
                mime="text/html",
            )
            if c3.button("Delete filtered logs", key="delete_filtered_logs"):
                removed = delete_user_activity(
                    indices=filtered["_idx"].astype(int).tolist()
                )
                log_admin_action(
                    st.session_state.get("auth_username", "admin"),
                    "delete_activity_logs",
                    details={"removed": removed},
                )
                st.warning(f"Deleted {removed} activity row(s).")
                st.rerun()

    with login_tab:
        st.markdown("#### Login activity")
        events = list(reversed(login_events))
        if events:
            st.dataframe(events, use_container_width=True, hide_index=True)
        else:
            st.info("No login activity recorded yet.")

    with audit_tab:
        st.markdown("#### Admin audit")
        audit_events = list(reversed(load_admin_audit()))
        if audit_events:
            st.dataframe(audit_events, use_container_width=True, hide_index=True)
        else:
            st.info("No admin audit actions recorded yet.")


def _render_workspace_sidebar() -> tuple[dict[str, Any] | None, str]:
    page_mode = "Workspace"
    with st.sidebar:
        st.markdown(f"**Signed in as:** {st.session_state['auth_username']}")
        st.caption(f"Role: {st.session_state['auth_role']}")
        if st.session_state.get("auth_role") == "admin":
            page_mode = st.radio(
                "Mode", ["Workspace", "Admin"], index=0, key="workspace_mode"
            )
        if st.button("Logout", key="logout_btn"):
            _logout()
            st.rerun()

    if page_mode == "Admin":
        return None, page_mode
    return render_sidebar(), page_mode


def _parse_page_range(raw: str) -> list[int] | None:
    text = (raw or "").strip()
    if not text:
        return None
    pages: set[int] = set()
    for part in text.split(","):
        part = part.strip()
        if not part:
            continue
        if "-" in part:
            a, b = part.split("-", 1)
            start = int(a.strip())
            end = int(b.strip())
            if end < start:
                start, end = end, start
            pages.update(range(start, end + 1))
        else:
            pages.add(int(part))
    return sorted(pages)


def _parse_json_schema(raw: str) -> dict[str, Any] | None:
    text = (raw or "").strip()
    if not text:
        return None
    try:
        parsed = json.loads(text)
    except json.JSONDecodeError as exc:
        st.error(f"Invalid JSON schema: {exc}")
        st.stop()
    if not isinstance(parsed, dict):
        st.error("JSON schema must be a JSON object.")
        st.stop()
    return parsed


def _build_pipelines(opts: dict[str, Any], di_model: str, cfg):
    pipelines = []
    if opts.get("enable_di"):
        pipelines.append(DocIntelligencePipeline(cfg, model_id=di_model))
    if opts.get("enable_hybrid_gpt54_mini"):
        pipelines.append(
            HybridDIPipeline(
                cfg,
                deployment=cfg.dep_gpt54_mini,
                model_key="gpt-5.4-mini",
                di_model_id=di_model,
                display_name=f"{di_model} + gpt-5.4-mini",
            )
        )
    if opts.get("enable_hybrid_gpt51"):
        pipelines.append(
            HybridDIPipeline(
                cfg,
                deployment=cfg.dep_gpt51,
                model_key="gpt-5.1",
                di_model_id=di_model,
                display_name=f"{di_model} + gpt-5.1",
            )
        )
    if opts.get("enable_hybrid_gpt4o_mini"):
        pipelines.append(
            HybridDIPipeline(
                cfg,
                deployment=cfg.dep_gpt4o_mini,
                model_key="gpt-4o-mini",
                di_model_id=di_model,
                display_name=f"{di_model} + gpt-4o-mini",
            )
        )
    if opts.get("enable_gpt5_vision"):
        pipelines.append(
            LLMVisionPipeline(
                cfg,
                deployment=cfg.dep_gpt5,
                model_key="gpt-5",
                display_name="gpt-5 (vision)",
            )
        )
    return pipelines


async def _run_judge(
    cfg, results: list[PipelineResult], gt_text: str | None
) -> dict[str, dict[str, Any]]:
    out: dict[str, dict[str, Any]] = {}
    if not gt_text:
        return out
    seen: set[str] = set()
    for r in results:
        if r.error or not r.raw_text or r.pipeline_id in seen:
            continue
        seen.add(r.pipeline_id)
        out[r.pipeline_id] = await judge_call(cfg, r.raw_text, gt_text)
    return out


def _snapshot_options(opts: dict[str, Any]) -> dict[str, Any]:
    """Return a JSON-safe copy of run options for the history record.

    Drops uploaded file handles and any value that is not JSON-serializable.
    """
    safe: dict[str, Any] = {}
    for key, value in opts.items():
        if key == "gt_file":
            continue
        if isinstance(value, (str, int, float, bool, type(None))):
            safe[key] = value
        elif isinstance(value, (list, dict)):
            try:
                json.dumps(value)
                safe[key] = value
            except (TypeError, ValueError):
                continue
    return safe


def _run_analysis(cfg, opts: dict[str, Any], uploaded) -> None:
    content = uploaded.read()
    mime = (
        uploaded.type
        or mimetypes.guess_type(uploaded.name)[0]
        or "application/octet-stream"
    )

    with st.spinner("Preprocessing pages..."):
        try:
            pages = preprocess(
                content,
                mime,
                deskew=opts["deskew"],
                denoise=opts["denoise"],
                grayscale=opts["grayscale"],
                page_range=_parse_page_range(opts["page_range_str"]),
            )
        except Exception as exc:  # noqa: BLE001
            st.error(f"Preprocessing failed: {exc}")
            st.stop()

    if not pages:
        st.error("No pages produced after preprocessing.")
        st.stop()

    st.success(f"Document prepared: {len(pages)} page(s).")
    with st.expander("Preview pages"):
        cols = st.columns(min(len(pages), 4))
        for i, p in enumerate(pages[:8]):
            cols[i % len(cols)].image(
                p, caption=f"Page {i + 1}", use_container_width=True
            )

    doc = DocumentInput(
        filename=uploaded.name, content=content, mime_type=mime, images=pages
    )
    pipelines = _build_pipelines(opts, opts["di_model"], cfg)
    if not pipelines:
        st.warning("Enable at least one pipeline in the sidebar.")
        st.stop()

    json_schema = (
        _parse_json_schema(opts["json_schema_text"])
        if opts["output_mode"] == "Strict JSON schema"
        else None
    )

    per_kwargs: dict[str, dict[str, Any]] = {}
    for p in pipelines:
        if p.id.startswith("llm-vision"):
            per_kwargs[p.id] = {
                "prompt": opts["extraction_prompt"],
                "temperature": opts["temperature"],
                "reasoning_effort": opts["reasoning_effort"],
                "json_schema": json_schema,
                "top_p": opts["top_p"],
                "seed": opts["seed"],
            }
        elif p.id.startswith("hybrid-di"):
            per_kwargs[p.id] = {
                "structuring_prompt": opts["structuring_prompt"],
                "temperature": opts["temperature"],
                "reasoning_effort": opts["reasoning_effort"],
                "json_schema": json_schema,
                "top_p": opts["top_p"],
                "seed": opts["seed"],
                "stitch_tables": opts["stitch_tables"],
                "normalize_numbers": opts["normalize_numbers"],
                "normalize_config": opts["normalize_config"],
            }

    repeat_n = int(opts.get("repeat_runs", 1))
    run_id = uuid4().hex
    current_user = st.session_state.get("auth_username", "")
    log_user_activity(
        current_user,
        "run_started",
        filename=uploaded.name,
        status="started",
        run_id=run_id,
        repeat_runs=repeat_n,
        pipeline_count=len(pipelines),
        details={"mime": mime},
    )

    label = (
        f"Running {len(pipelines)} pipeline(s) x {repeat_n} run(s) in parallel..."
        if repeat_n > 1
        else f"Running {len(pipelines)} pipeline(s) in parallel..."
    )

    try:
        with st.spinner(label):
            results = asyncio.run(
                run_all(pipelines, doc, per_kwargs, repeat_n=repeat_n)
            )
    except Exception as exc:  # noqa: BLE001
        log_user_activity(
            current_user,
            "run_completed",
            filename=uploaded.name,
            status="failed",
            run_id=run_id,
            repeat_runs=repeat_n,
            pipeline_count=len(pipelines),
            details={"error": str(exc)},
        )
        st.error(f"Analysis failed: {exc}")
        st.stop()

    log_user_activity(
        current_user,
        "run_completed",
        filename=uploaded.name,
        status="success",
        run_id=run_id,
        repeat_runs=repeat_n,
        pipeline_count=len(pipelines),
        details={"result_count": len(results)},
    )

    gt_text: str | None = None
    gt_json: dict[str, Any] | None = None
    if opts["gt_file"] is not None:
        gt_bytes = opts["gt_file"].read()
        try:
            gt_decoded = gt_bytes.decode("utf-8")
        except UnicodeDecodeError:
            gt_decoded = gt_bytes.decode("latin-1", errors="ignore")
        if opts["gt_file"].name.lower().endswith(".json"):
            try:
                gt_json = json.loads(gt_decoded)
                gt_text = json.dumps(gt_json, indent=2)
            except json.JSONDecodeError:
                gt_text = gt_decoded
        else:
            gt_text = gt_decoded

    judge_scores: dict[str, dict[str, Any]] | None = None
    if opts["run_judge"]:
        with st.spinner("Running LLM-as-judge..."):
            judge_scores = asyncio.run(_run_judge(cfg, results, gt_text))

    st.session_state["last_results"] = results
    st.session_state["last_gt_text"] = gt_text
    st.session_state["last_gt_json"] = gt_json
    st.session_state["last_judge_scores"] = judge_scores
    st.session_state["last_uploaded_name"] = uploaded.name
    st.session_state["doctalk_history"] = []

    try:
        save_run(
            {
                "run_id": run_id,
                "username": current_user,
                "filename": uploaded.name,
                "mime_type": mime,
                "pages": len(pages),
                "repeat_runs": repeat_n,
                "pipeline_count": len(pipelines),
                "redacted": False,
                "gt_present": gt_text is not None,
                "judge_present": judge_scores is not None,
                "total_cost_usd": sum(
                    float(getattr(r, "cost_usd", 0.0) or 0.0) for r in results
                ),
                "options": _snapshot_options(opts),
            },
            results,
        )
    except Exception as exc:  # noqa: BLE001
        st.warning(f"Run history could not be saved: {exc}")


def _open_run(run_id: str) -> None:
    """Load a stored run into session state and switch to the Compare view."""
    meta, results = load_run(run_id)
    if not results:
        st.warning("That run could not be loaded (it may have been deleted).")
        return
    st.session_state["last_results"] = results
    st.session_state["last_gt_text"] = None
    st.session_state["last_gt_json"] = None
    st.session_state["last_judge_scores"] = None
    st.session_state["last_uploaded_name"] = (
        f"{meta.get('filename', 'run')} (from history)"
    )
    st.session_state["doctalk_history"] = []
    st.session_state["active_top_nav"] = "Compare"
    st.rerun()


def _render_history(cfg) -> None:
    current_user = st.session_state.get("auth_username", "")
    runs = list_runs(username=current_user, limit=100)
    if not runs:
        st.info("No saved runs yet. Run an analysis to build your history.")
        return

    st.caption(f"Showing your {len(runs)} most recent run(s).")
    table = [
        {
            "Date (UTC)": (r.get("created_utc") or "")[:19].replace("T", " "),
            "File": r.get("filename", ""),
            "Pipelines": r.get("pipeline_count"),
            "Repeat": r.get("repeat_runs"),
            "Cost (USD)": round(float(r.get("total_cost_usd") or 0.0), 4),
            "Redacted": "Yes" if r.get("redacted") else "No",
            "GT": "Yes" if r.get("gt_present") else "No",
            "Judge": "Yes" if r.get("judge_present") else "No",
        }
        for r in runs
    ]
    st.dataframe(pd.DataFrame(table), use_container_width=True, hide_index=True)

    for r in runs:
        run_id = r["run_id"]
        label = (
            f"{(r.get('created_utc') or '')[:19].replace('T', ' ')} "
            f"\u2014 {r.get('filename', '')}"
        )
        with st.container(border=True):
            cols = st.columns([6, 1, 1])
            cols[0].markdown(f"**{label}**")
            if cols[1].button("Open", key=f"open_{run_id}"):
                _open_run(run_id)
            if cols[2].button("Delete", key=f"del_{run_id}"):
                delete_run(run_id)
                st.rerun()


def _render_workspace(cfg, opts: dict[str, Any], active_top_nav: str) -> None:
    if active_top_nav == "History":
        _render_history(cfg)
        return

    uploaded = st.file_uploader(
        "Upload a document",
        type=["pdf", "png", "jpg", "jpeg", "tif", "tiff"],
        accept_multiple_files=False,
    )
    run_btn = st.button("Start analysis", type="primary", disabled=uploaded is None)

    if run_btn and uploaded is not None:
        _run_analysis(cfg, opts, uploaded)

    if active_top_nav == "Insights":
        st.info(
            "Insights is reserved for a richer summary workspace. For now, use Compare for document runs and Determinism for repeat-run analysis."
        )
        return

    if active_top_nav == "Determinism":
        st.info(
            "Determinism mode focuses on repeat-run stability. Run the same document with repeated runs >= 2, then inspect the Determinism tab in the results below."
        )

    results = st.session_state.get("last_results")
    if results:
        if st.session_state.get("last_uploaded_name"):
            st.caption(
                f"Showing latest results for {st.session_state['last_uploaded_name']}"
            )
        render_results(
            results,
            st.session_state.get("last_gt_text"),
            st.session_state.get("last_gt_json"),
            st.session_state.get("last_judge_scores"),
            cfg=cfg,
            show_ai_summary=opts["show_ai_summary"],
        )
    elif uploaded is None:
        st.info(
            "Upload a PDF or image to begin. Configure pipelines and options in the sidebar."
        )


ensure_storage()
_init_session_state()

if not st.session_state["authenticated"]:
    _render_login_page()
else:
    opts, mode = _render_workspace_sidebar()
    if mode == "Admin":
        _render_admin_page()
    elif opts is not None:
        _render_app_header()
        active_top_nav = _render_top_nav()
        cfg = load_config()
        _render_workspace(cfg, opts, active_top_nav)

st.markdown(
    """
<div class="app-footer">
  <strong>Smart OCR for Serious Documents</strong> — Demo MVP. Decisions are advisory and require human review. · <a href="mailto:Kasm.Shaikh@microsoft.com?subject=OCR%20Demo%20Support%20Request">Connect Kasam Shaikh</a>
</div>
""",
    unsafe_allow_html=True,
)
