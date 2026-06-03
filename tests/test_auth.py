import tempfile
import unittest
from pathlib import Path

import src.auth as auth


class AuthStorageTests(unittest.TestCase):
    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        root = Path(self.temp_dir.name)
        self.original_paths = (
            auth.DATA_DIR,
            auth.USERS_FILE,
            auth.REQUESTS_FILE,
            auth.LOGIN_ACTIVITY_FILE,
            auth.USER_ACTIVITY_FILE,
            auth.ADMIN_AUDIT_FILE,
        )
        auth.DATA_DIR = root / "data"
        auth.USERS_FILE = auth.DATA_DIR / "users.json"
        auth.REQUESTS_FILE = auth.DATA_DIR / "passcode_requests.json"
        auth.LOGIN_ACTIVITY_FILE = auth.DATA_DIR / "login_activity.json"
        auth.USER_ACTIVITY_FILE = auth.DATA_DIR / "user_activity.json"
        auth.ADMIN_AUDIT_FILE = auth.DATA_DIR / "admin_audit.json"

    def tearDown(self):
        (
            auth.DATA_DIR,
            auth.USERS_FILE,
            auth.REQUESTS_FILE,
            auth.LOGIN_ACTIVITY_FILE,
            auth.USER_ACTIVITY_FILE,
            auth.ADMIN_AUDIT_FILE,
        ) = self.original_paths
        self.temp_dir.cleanup()

    def test_ensure_storage_bootstraps_admin(self):
        auth.ensure_storage()
        users = auth.load_users()
        self.assertEqual(len(users), 1)
        self.assertEqual(users[0]["username"], auth.DEFAULT_ADMIN_USERNAME)
        self.assertEqual(users[0]["role"], "admin")

    def test_authenticate_user_accepts_default_admin_passcode(self):
        auth.ensure_storage()
        user = auth.authenticate_user(
            auth.DEFAULT_ADMIN_USERNAME, auth.DEFAULT_ADMIN_PASSCODE
        )
        self.assertIsNotNone(user)
        self.assertEqual(user["role"], "admin")

    def test_passcode_request_and_login_log_persist(self):
        auth.ensure_storage()
        auth.submit_passcode_request("alice", "alice@example.com", "Need access")
        auth.log_login_attempt("alice", False, reason="invalid_credentials")
        requests = auth.load_passcode_requests()
        events = auth.load_login_activity()
        self.assertEqual(len(requests), 1)
        self.assertEqual(requests[0]["name"], "alice")
        self.assertEqual(len(events), 1)
        self.assertFalse(events[0]["success"])
        self.assertEqual(events[0]["reason"], "invalid_credentials")

    def test_approve_passcode_request_creates_user_and_updates_request(self):
        auth.ensure_storage()
        auth.submit_passcode_request("Tanay", "tanay@example.com", "Need access")

        created = auth.approve_passcode_request(0, "kasam", "Tanay@123")

        users = auth.load_users()
        requests = auth.load_passcode_requests()
        self.assertEqual(created["username"], "tanay")
        self.assertEqual(len(users), 2)
        self.assertEqual(requests[0]["status"], "approved")
        self.assertEqual(requests[0]["approved_by"], "kasam")
        self.assertEqual(requests[0]["granted_username"], "tanay")
        self.assertIsNotNone(auth.authenticate_user("tanay", "Tanay@123"))

    def test_approve_passcode_request_rejects_duplicate_username(self):
        auth.ensure_storage()
        auth.create_user("tanay", "Temp@123")
        auth.submit_passcode_request("Tanay", "tanay@example.com", "Need access")

        with self.assertRaises(ValueError):
            auth.approve_passcode_request(0, "kasam", "Other@123")

    def test_approve_passcode_request_requires_non_empty_passcode(self):
        auth.ensure_storage()
        auth.submit_passcode_request("Alice", "alice@example.com", "Need access")

        with self.assertRaises(ValueError):
            auth.approve_passcode_request(0, "kasam", "")

    def test_deny_passcode_request_marks_request_denied(self):
        auth.ensure_storage()
        auth.submit_passcode_request("Bob", "bob@example.com", "Need access")

        auth.deny_passcode_request(0, "kasam", "not eligible")

        requests = auth.load_passcode_requests()
        self.assertEqual(requests[0]["status"], "denied")
        self.assertEqual(requests[0]["denied_by"], "kasam")
        self.assertEqual(requests[0]["denial_reason"], "not eligible")

    def test_update_user_role_and_passcode(self):
        auth.ensure_storage()
        auth.create_user("alice", "Old@123", role="user")

        updated = auth.update_user(
            "alice",
            role="admin",
            new_passcode="New@123",
            updated_by="kasam",
        )

        self.assertEqual(updated["role"], "admin")
        self.assertEqual(updated["updated_by"], "kasam")
        self.assertIsNotNone(auth.authenticate_user("alice", "New@123"))

    def test_soft_delete_user_blocks_auth(self):
        auth.ensure_storage()
        auth.create_user("charlie", "Temp@123", role="user")

        auth.soft_delete_user("charlie", "kasam")

        self.assertIsNone(auth.authenticate_user("charlie", "Temp@123"))

    def test_delete_user_activity_by_indices(self):
        auth.ensure_storage()
        auth.log_user_activity(
            "kasam", "run_started", filename="a.pdf", status="started"
        )
        auth.log_user_activity(
            "kasam", "run_completed", filename="a.pdf", status="success"
        )

        removed = auth.delete_user_activity(indices=[0])
        events = auth.load_user_activity()

        self.assertEqual(removed, 1)
        self.assertEqual(len(events), 1)
        self.assertEqual(events[0]["action"], "run_completed")

    def test_deny_passcode_request_marks_denied(self):
        auth.ensure_storage()
        auth.submit_passcode_request("Alice", "alice@example.com", "Need access")

        auth.deny_passcode_request(0, "kasam", "Insufficient details")

        requests = auth.load_passcode_requests()
        self.assertEqual(requests[0]["status"], "denied")
        self.assertEqual(requests[0]["denied_by"], "kasam")
        self.assertEqual(requests[0]["denial_reason"], "Insufficient details")

    def test_update_user_role_and_passcode(self):
        auth.ensure_storage()
        auth.create_user("alice", "Old@123", role="user")

        updated = auth.update_user(
            "alice", role="admin", new_passcode="New@123", updated_by="kasam"
        )

        self.assertEqual(updated["role"], "admin")
        self.assertTrue(auth.authenticate_user("alice", "New@123"))
        self.assertIsNone(auth.authenticate_user("alice", "Old@123"))

    def test_soft_delete_disables_login(self):
        auth.ensure_storage()
        auth.create_user("alice", "Old@123", role="user")

        deleted = auth.soft_delete_user("alice", "kasam")
        self.assertFalse(deleted["active"])
        self.assertIsNone(auth.authenticate_user("alice", "Old@123"))

    def test_user_activity_append_and_delete(self):
        auth.ensure_storage()
        auth.log_user_activity("kasam", "run_started", filename="a.pdf", status="ok")
        auth.log_user_activity("kasam", "run_completed", filename="a.pdf", status="ok")

        events = auth.load_user_activity()
        self.assertEqual(len(events), 2)

        removed = auth.delete_user_activity(indices=[0])
        self.assertEqual(removed, 1)
        self.assertEqual(len(auth.load_user_activity()), 1)


if __name__ == "__main__":
    unittest.main()
