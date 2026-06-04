"""Tests for the SQLite run-history store."""

from __future__ import annotations

import importlib
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

from src.pipelines.base import PipelineResult


class RunStoreTests(unittest.TestCase):
    def setUp(self) -> None:
        self._tmp = TemporaryDirectory()
        self._db_path = Path(self._tmp.name) / "runs.db"
        import os

        os.environ["RUNS_DB_PATH"] = str(self._db_path)
        # Reload module so it picks up the env var at call time (it reads lazily,
        # but reload keeps tests isolated from any prior import state).
        from src.storage import run_store

        self.store = importlib.reload(run_store)

    def tearDown(self) -> None:
        import os

        os.environ.pop("RUNS_DB_PATH", None)
        self._tmp.cleanup()

    def _sample_results(self) -> list[PipelineResult]:
        return [
            PipelineResult(
                pipeline_id="llm-vision-gpt4o",
                display_name="GPT-4o vision",
                raw_text="Booking fee 4.00",
                structured_json={"booking_fee": "4.00"},
                cost_usd=0.0123,
                run_index=1,
                raw_response={"not": "json-safe"},
            ),
            PipelineResult(
                pipeline_id="di-only",
                display_name="DI only",
                raw_text="Booking fee 4.00",
                cost_usd=0.002,
                run_index=1,
            ),
        ]

    def _meta(self, run_id: str = "run-1", username: str = "alice") -> dict:
        return {
            "run_id": run_id,
            "username": username,
            "filename": "invoice.pdf",
            "mime_type": "application/pdf",
            "pages": 2,
            "repeat_runs": 1,
            "pipeline_count": 2,
            "redacted": False,
            "gt_present": False,
            "judge_present": False,
            "options": {"di_model": "prebuilt-layout", "temperature": 0.0},
        }

    def test_save_and_load_round_trip(self) -> None:
        results = self._sample_results()
        self.store.save_run(self._meta(), results)

        meta, loaded = self.store.load_run("run-1")
        self.assertIsNotNone(meta)
        self.assertEqual(len(loaded), 2)
        self.assertEqual(loaded[0].pipeline_id, "llm-vision-gpt4o")
        self.assertEqual(loaded[0].structured_json, {"booking_fee": "4.00"})
        self.assertEqual(loaded[0].run_index, 1)
        # raw_response must be dropped (not serializable / not needed for replay).
        self.assertIsNone(loaded[0].raw_response)
        self.assertEqual(
            meta["options"], {"di_model": "prebuilt-layout", "temperature": 0.0}
        )

    def test_total_cost_computed_when_absent(self) -> None:
        meta = self._meta()
        meta.pop("total_cost_usd", None)
        self.store.save_run(meta, self._sample_results())
        rows = self.store.list_runs(username="alice")
        self.assertEqual(len(rows), 1)
        self.assertAlmostEqual(rows[0]["total_cost_usd"], 0.0143, places=4)

    def test_list_runs_filters_by_user_and_orders_desc(self) -> None:
        self.store.save_run(
            {**self._meta("a", "alice"), "created_utc": "2026-01-01T00:00:00"},
            self._sample_results(),
        )
        self.store.save_run(
            {**self._meta("b", "alice"), "created_utc": "2026-02-01T00:00:00"},
            self._sample_results(),
        )
        self.store.save_run(
            {**self._meta("c", "bob"), "created_utc": "2026-03-01T00:00:00"},
            self._sample_results(),
        )

        alice_runs = self.store.list_runs(username="alice")
        self.assertEqual([r["run_id"] for r in alice_runs], ["b", "a"])
        self.assertEqual(len(self.store.list_runs(username="bob")), 1)
        self.assertEqual(len(self.store.list_runs()), 3)

    def test_delete_run_cascades(self) -> None:
        self.store.save_run(self._meta(), self._sample_results())
        self.store.delete_run("run-1")
        meta, loaded = self.store.load_run("run-1")
        self.assertIsNone(meta)
        self.assertEqual(loaded, [])

    def test_save_is_idempotent_on_same_run_id(self) -> None:
        self.store.save_run(self._meta(), self._sample_results())
        self.store.save_run(self._meta(), self._sample_results()[:1])
        _, loaded = self.store.load_run("run-1")
        self.assertEqual(len(loaded), 1)
        self.assertEqual(len(self.store.list_runs(username="alice")), 1)


if __name__ == "__main__":
    unittest.main()
