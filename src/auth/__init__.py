from __future__ import annotations

import base64
import hashlib
import hmac
import json
import os
import re
import secrets
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable

try:
    from azure.core import MatchConditions
    from azure.core.exceptions import ResourceModifiedError, ResourceNotFoundError
    from azure.identity import DefaultAzureCredential
    from azure.storage.blob import BlobServiceClient
except ImportError:  # pragma: no cover - optional dependency in local runs
    MatchConditions = None  # type: ignore[assignment]
    ResourceModifiedError = Exception  # type: ignore[assignment]
    ResourceNotFoundError = Exception  # type: ignore[assignment]
    DefaultAzureCredential = None  # type: ignore[assignment]
    BlobServiceClient = None  # type: ignore[assignment]

ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = ROOT / "data"
USERS_FILE = DATA_DIR / "users.json"
REQUESTS_FILE = DATA_DIR / "passcode_requests.json"
LOGIN_ACTIVITY_FILE = DATA_DIR / "login_activity.json"
USER_ACTIVITY_FILE = DATA_DIR / "user_activity.json"
ADMIN_AUDIT_FILE = DATA_DIR / "admin_audit.json"

AUTH_STORAGE_BACKEND = os.getenv("AUTH_STORAGE_BACKEND", "local").strip().lower()
AUTH_BLOB_CONNECTION_STRING = os.getenv("AUTH_BLOB_CONNECTION_STRING", "").strip()
AUTH_BLOB_ACCOUNT_URL = os.getenv("AUTH_BLOB_ACCOUNT_URL", "").strip()
AUTH_BLOB_CONTAINER = os.getenv("AUTH_BLOB_CONTAINER", "appdata").strip() or "appdata"
AUTH_BLOB_PREFIX = os.getenv("AUTH_BLOB_PREFIX", "auth").strip().strip("/")

DEFAULT_ADMIN_USERNAME = os.getenv("DEFAULT_ADMIN_USERNAME", "admin").strip() or "admin"
DEFAULT_ADMIN_PASSCODE = os.getenv("DEFAULT_ADMIN_PASSCODE", "ChangeMe123!")

_BLOB_CONTAINER_CLIENT: Any | None = None


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _pbkdf2_hash(passcode: str, salt: str | None = None) -> str:
    raw_salt = salt or secrets.token_hex(16)
    derived = hashlib.pbkdf2_hmac(
        "sha256", passcode.encode("utf-8"), raw_salt.encode("utf-8"), 120_000
    )
    encoded = base64.b64encode(derived).decode("ascii")
    return f"pbkdf2_sha256${raw_salt}${encoded}"


def _verify_passcode(passcode: str, stored_hash: str) -> bool:
    try:
        algorithm, salt, _expected = stored_hash.split("$", 2)
    except ValueError:
        return False
    if algorithm != "pbkdf2_sha256":
        return False
    actual = _pbkdf2_hash(passcode, salt)
    return hmac.compare_digest(actual, stored_hash)


def _normalize_username(candidate: str) -> str:
    lowered = candidate.strip().lower()
    cleaned = re.sub(r"[^a-z0-9._-]", "", lowered)
    return cleaned


def _blob_enabled() -> bool:
    return (
        AUTH_STORAGE_BACKEND == "blob"
        and (bool(AUTH_BLOB_CONNECTION_STRING) or bool(AUTH_BLOB_ACCOUNT_URL))
        and BlobServiceClient is not None
    )


def _blob_name(path: Path) -> str:
    if AUTH_BLOB_PREFIX:
        return f"{AUTH_BLOB_PREFIX}/{path.name}"
    return path.name


def _get_blob_container_client() -> Any | None:
    global _BLOB_CONTAINER_CLIENT
    if not _blob_enabled():
        return None
    if _BLOB_CONTAINER_CLIENT is not None:
        return _BLOB_CONTAINER_CLIENT

    if AUTH_BLOB_CONNECTION_STRING:
        service = BlobServiceClient.from_connection_string(AUTH_BLOB_CONNECTION_STRING)
    else:
        if DefaultAzureCredential is None:
            return None
        credential = DefaultAzureCredential(
            exclude_interactive_browser_credential=False
        )
        service = BlobServiceClient(
            account_url=AUTH_BLOB_ACCOUNT_URL, credential=credential
        )
    container = service.get_container_client(AUTH_BLOB_CONTAINER)
    if not container.exists():
        container.create_container()
    _BLOB_CONTAINER_CLIENT = container
    return _BLOB_CONTAINER_CLIENT


def _write_json(path: Path, payload: Any) -> None:
    text = json.dumps(payload, indent=2)
    container = _get_blob_container_client()
    if container is not None:
        container.upload_blob(
            name=_blob_name(path),
            data=text.encode("utf-8"),
            overwrite=True,
        )
        return
    path.write_text(text, encoding="utf-8")


def _update_json(path: Path, default: Any, updater: Callable[[Any], Any]) -> Any:
    container = _get_blob_container_client()
    if container is None:
        current = _read_json(path, default)
        updated = updater(current)
        _write_json(path, updated)
        return updated

    blob_client = container.get_blob_client(_blob_name(path))
    attempts = 5
    for _ in range(attempts):
        exists = True
        etag = None
        try:
            raw = blob_client.download_blob().readall().decode("utf-8")
            props = blob_client.get_blob_properties()
            etag = props.etag
            try:
                current = json.loads(raw)
            except json.JSONDecodeError:
                current = default
        except ResourceNotFoundError:
            exists = False
            current = default

        updated = updater(current)
        data = json.dumps(updated, indent=2).encode("utf-8")
        try:
            if exists and etag and MatchConditions is not None:
                blob_client.upload_blob(
                    data,
                    overwrite=True,
                    etag=etag,
                    match_condition=MatchConditions.IfNotModified,
                )
            elif exists:
                blob_client.upload_blob(data, overwrite=True)
            else:
                blob_client.upload_blob(data, overwrite=False)
            return updated
        except ResourceModifiedError:
            continue

    raise RuntimeError(f"Concurrent update retries exceeded for {path.name}")


def _read_json(path: Path, default: Any) -> Any:
    container = _get_blob_container_client()
    if container is not None:
        blob_client = container.get_blob_client(_blob_name(path))
        if not blob_client.exists():
            return default
        try:
            data = blob_client.download_blob().readall().decode("utf-8")
            return json.loads(data)
        except (UnicodeDecodeError, json.JSONDecodeError):
            return default

    if not path.exists():
        return default
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return default


def ensure_storage() -> None:
    if _get_blob_container_client() is None:
        DATA_DIR.mkdir(parents=True, exist_ok=True)

    users_payload = _read_json(USERS_FILE, {"users": []})
    users = list(users_payload.get("users", []))
    needs_bootstrap = not users or any(
        str(user.get("passcode_hash", "")).startswith("pbkdf2_sha256$bootstrap$")
        for user in users
    )
    if needs_bootstrap:
        _write_json(
            USERS_FILE,
            {
                "users": [
                    {
                        "username": DEFAULT_ADMIN_USERNAME,
                        "passcode_hash": _pbkdf2_hash(DEFAULT_ADMIN_PASSCODE),
                        "role": "admin",
                        "created_at": _utc_now(),
                    }
                ]
            },
        )

    if _read_json(REQUESTS_FILE, None) is None:
        _write_json(REQUESTS_FILE, {"requests": []})
    if _read_json(LOGIN_ACTIVITY_FILE, None) is None:
        _write_json(LOGIN_ACTIVITY_FILE, {"events": []})
    if _read_json(USER_ACTIVITY_FILE, None) is None:
        _write_json(USER_ACTIVITY_FILE, {"events": []})
    if _read_json(ADMIN_AUDIT_FILE, None) is None:
        _write_json(ADMIN_AUDIT_FILE, {"events": []})


def load_users() -> list[dict[str, Any]]:
    ensure_storage()
    payload = _read_json(USERS_FILE, {"users": []})
    return list(payload.get("users", []))


def create_user(username: str, passcode: str, *, role: str = "user") -> dict[str, Any]:
    ensure_storage()
    normalized = _normalize_username(username)
    if not normalized:
        raise ValueError("Username must contain letters or numbers.")

    record: dict[str, Any] = {}

    def _mutate(payload: Any) -> Any:
        users = list(payload.get("users", []))
        exists = any(
            str(user.get("username", "")).lower() == normalized for user in users
        )
        if exists:
            raise ValueError("Username already exists.")

        nonlocal record
        record = {
            "username": normalized,
            "passcode_hash": _pbkdf2_hash(passcode),
            "role": role if role in {"user", "admin"} else "user",
            "active": True,
            "created_at": _utc_now(),
        }
        users.append(record)
        payload["users"] = users
        return payload

    _update_json(USERS_FILE, {"users": []}, _mutate)
    return record


def authenticate_user(username: str, passcode: str) -> dict[str, Any] | None:
    normalized = username.strip().lower()
    for user in load_users():
        if str(user.get("username", "")).lower() != normalized:
            continue
        if not bool(user.get("active", True)):
            return None
        if _verify_passcode(passcode, str(user.get("passcode_hash", ""))):
            return user
        return None
    return None


def log_login_attempt(
    username: str, success: bool, *, role: str = "", reason: str = ""
) -> None:
    ensure_storage()

    def _mutate(payload: Any) -> Any:
        payload.setdefault("events", []).append(
            {
                "username": username.strip(),
                "success": success,
                "role": role,
                "reason": reason,
                "timestamp": _utc_now(),
            }
        )
        return payload

    _update_json(LOGIN_ACTIVITY_FILE, {"events": []}, _mutate)


def load_login_activity() -> list[dict[str, Any]]:
    ensure_storage()
    payload = _read_json(LOGIN_ACTIVITY_FILE, {"events": []})
    return list(payload.get("events", []))


def submit_passcode_request(name: str, contact: str, reason: str) -> None:
    ensure_storage()

    def _mutate(payload: Any) -> Any:
        payload.setdefault("requests", []).append(
            {
                "name": name.strip(),
                "contact": contact.strip(),
                "reason": reason.strip(),
                "status": "pending",
                "requested_at": _utc_now(),
            }
        )
        return payload

    _update_json(REQUESTS_FILE, {"requests": []}, _mutate)


def load_passcode_requests() -> list[dict[str, Any]]:
    ensure_storage()
    payload = _read_json(REQUESTS_FILE, {"requests": []})
    return list(payload.get("requests", []))


def approve_passcode_request(
    request_index: int,
    approved_by: str,
    passcode: str,
    *,
    username_override: str = "",
    role: str = "user",
) -> dict[str, str]:
    ensure_storage()
    if not passcode.strip():
        raise ValueError("Passcode is required.")

    selected_username = ""

    def _resolve(payload: Any) -> Any:
        requests = list(payload.get("requests", []))

        if request_index < 0 or request_index >= len(requests):
            raise IndexError("Invalid passcode request index.")

        target = requests[request_index]
        if str(target.get("status", "pending")).lower() != "pending":
            raise ValueError("Request has already been processed.")

        raw_name = str(target.get("name", "")).strip()
        requested_username = username_override.strip() or raw_name
        username = _normalize_username(requested_username)
        if not username:
            raise ValueError("Unable to derive username for this request.")

        nonlocal selected_username
        selected_username = username
        return payload

    _resolve(_read_json(REQUESTS_FILE, {"requests": []}))
    create_user(selected_username, passcode, role=role)

    def _mark_approved(payload: Any) -> Any:
        requests = list(payload.get("requests", []))
        if request_index < 0 or request_index >= len(requests):
            raise IndexError("Invalid passcode request index.")
        target = requests[request_index]
        if str(target.get("status", "pending")).lower() != "pending":
            raise ValueError("Request has already been processed.")

        target["status"] = "approved"
        target["approved_by"] = approved_by.strip()
        target["approved_at"] = _utc_now()
        target["granted_username"] = selected_username
        requests[request_index] = target
        payload["requests"] = requests
        return payload

    _update_json(REQUESTS_FILE, {"requests": []}, _mark_approved)
    log_admin_action(
        approved_by.strip() or "admin",
        "approve_request",
        target_username=selected_username,
        details={"request_index": request_index, "role": role},
    )
    return {"username": selected_username}


def deny_passcode_request(request_index: int, denied_by: str, reason: str = "") -> None:
    ensure_storage()

    def _mutate(payload: Any) -> Any:
        requests = list(payload.get("requests", []))
        if request_index < 0 or request_index >= len(requests):
            raise IndexError("Invalid passcode request index.")

        target = requests[request_index]
        if str(target.get("status", "pending")).lower() != "pending":
            raise ValueError("Request has already been processed.")

        target["status"] = "denied"
        target["denied_by"] = denied_by.strip()
        target["denied_at"] = _utc_now()
        target["denial_reason"] = reason.strip()
        requests[request_index] = target
        payload["requests"] = requests
        return payload

    _update_json(REQUESTS_FILE, {"requests": []}, _mutate)
    log_admin_action(
        denied_by.strip() or "admin",
        "deny_request",
        details={"request_index": request_index, "reason": reason.strip()},
    )


def update_user(
    username: str,
    *,
    role: str | None = None,
    new_passcode: str | None = None,
    active: bool | None = None,
    updated_by: str = "",
) -> dict[str, Any]:
    ensure_storage()
    normalized = _normalize_username(username)
    if not normalized:
        raise ValueError("Invalid username.")
    if role is not None and role not in {"user", "admin"}:
        raise ValueError("Role must be 'user' or 'admin'.")
    if new_passcode is not None and not new_passcode.strip():
        raise ValueError("Passcode cannot be empty.")

    updated_record: dict[str, Any] = {}

    def _mutate(payload: Any) -> Any:
        users = list(payload.get("users", []))
        for idx, user in enumerate(users):
            if str(user.get("username", "")).lower() != normalized:
                continue

            if role is not None:
                user["role"] = role
            if new_passcode is not None:
                user["passcode_hash"] = _pbkdf2_hash(new_passcode)
            if active is not None:
                user["active"] = active
                if not active:
                    user["deleted_at"] = _utc_now()
                    user["deleted_by"] = updated_by.strip()
                else:
                    user.pop("deleted_at", None)
                    user.pop("deleted_by", None)

            user["updated_at"] = _utc_now()
            user["updated_by"] = updated_by.strip()
            users[idx] = user

            nonlocal updated_record
            updated_record = dict(user)
            payload["users"] = users
            return payload

        raise ValueError("User not found.")

    _update_json(USERS_FILE, {"users": []}, _mutate)
    return updated_record


def soft_delete_user(username: str, deleted_by: str = "") -> dict[str, Any]:
    normalized = _normalize_username(username)
    users = load_users()
    active_admins = [
        u
        for u in users
        if bool(u.get("active", True)) and str(u.get("role", "user")) == "admin"
    ]
    target = next(
        (
            u
            for u in users
            if str(u.get("username", "")).lower() == normalized
            and bool(u.get("active", True))
        ),
        None,
    )
    if target is None:
        raise ValueError("Active user not found.")
    if target.get("role") == "admin" and len(active_admins) <= 1:
        raise ValueError("Cannot delete the last active admin.")

    updated = update_user(normalized, active=False, updated_by=deleted_by)
    log_admin_action(
        deleted_by.strip() or "admin",
        "delete_user",
        target_username=normalized,
    )
    return updated


def list_users(*, include_inactive: bool = True) -> list[dict[str, Any]]:
    users = load_users()
    if include_inactive:
        return users
    return [u for u in users if bool(u.get("active", True))]


def log_user_activity(
    username: str,
    action: str,
    *,
    filename: str = "",
    status: str = "",
    run_id: str = "",
    repeat_runs: int = 0,
    pipeline_count: int = 0,
    details: dict[str, Any] | None = None,
) -> None:
    ensure_storage()

    def _mutate(payload: Any) -> Any:
        payload.setdefault("events", []).append(
            {
                "username": username.strip(),
                "action": action.strip(),
                "filename": filename.strip(),
                "status": status.strip(),
                "run_id": run_id.strip(),
                "repeat_runs": int(repeat_runs or 0),
                "pipeline_count": int(pipeline_count or 0),
                "details": details or {},
                "timestamp": _utc_now(),
            }
        )
        return payload

    _update_json(USER_ACTIVITY_FILE, {"events": []}, _mutate)


def load_user_activity() -> list[dict[str, Any]]:
    ensure_storage()
    payload = _read_json(USER_ACTIVITY_FILE, {"events": []})
    return list(payload.get("events", []))


def delete_user_activity(
    *,
    indices: list[int] | None = None,
    before_iso: str = "",
) -> int:
    ensure_storage()
    removed = 0
    index_set = set(indices or [])

    before_dt = None
    if before_iso.strip():
        before_dt = datetime.fromisoformat(before_iso.strip().replace("Z", "+00:00"))

    def _mutate(payload: Any) -> Any:
        events = list(payload.get("events", []))
        keep: list[dict[str, Any]] = []
        nonlocal removed

        for idx, event in enumerate(events):
            drop = False
            if indices is not None and idx in index_set:
                drop = True
            if before_dt is not None:
                ts_raw = str(event.get("timestamp", "")).strip()
                try:
                    ts_dt = datetime.fromisoformat(ts_raw.replace("Z", "+00:00"))
                    if ts_dt <= before_dt:
                        drop = True
                except ValueError:
                    pass

            if drop:
                removed += 1
            else:
                keep.append(event)

        payload["events"] = keep
        return payload

    _update_json(USER_ACTIVITY_FILE, {"events": []}, _mutate)
    return removed


def log_admin_action(
    admin_username: str,
    action: str,
    *,
    target_username: str = "",
    details: dict[str, Any] | None = None,
) -> None:
    ensure_storage()

    def _mutate(payload: Any) -> Any:
        payload.setdefault("events", []).append(
            {
                "admin_username": admin_username.strip(),
                "action": action.strip(),
                "target_username": target_username.strip(),
                "details": details or {},
                "timestamp": _utc_now(),
            }
        )
        return payload

    _update_json(ADMIN_AUDIT_FILE, {"events": []}, _mutate)


def load_admin_audit() -> list[dict[str, Any]]:
    ensure_storage()
    payload = _read_json(ADMIN_AUDIT_FILE, {"events": []})
    return list(payload.get("events", []))


def login_summary() -> dict[str, Any]:
    events = load_login_activity()
    success_rows = [event for event in events if event.get("success")]
    unique_users = sorted(
        {
            str(event.get("username") or "")
            for event in success_rows
            if event.get("username")
        }
    )
    return {
        "successful_logins": len(success_rows),
        "unique_users": len(unique_users),
        "unique_usernames": unique_users,
    }
