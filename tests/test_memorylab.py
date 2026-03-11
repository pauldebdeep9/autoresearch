import json
import tempfile
import unittest
from datetime import datetime, timezone
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

import memorylab


class MemoryLabTests(unittest.TestCase):
    def setUp(self):
        self.tempdir = tempfile.TemporaryDirectory()
        self.root = Path(self.tempdir.name)
        self.memorylab_dir = self.root / "results" / "memorylab"
        self.patcher = mock.patch.multiple(
            memorylab,
            ROOT=self.root,
            RESULTS_TSV=self.root / "results.tsv",
            MEMORYLAB_DIR=self.memorylab_dir,
            LEDGER_PATH=self.memorylab_dir / "experiments.jsonl",
            REGISTRY_PATH=self.memorylab_dir / "champion_challenger.json",
            REPORTS_DIR=self.memorylab_dir / "reports",
            RUNS_DIR=self.memorylab_dir / "runs",
        )
        self.patcher.start()
        self.addCleanup(self.patcher.stop)
        self.addCleanup(self.tempdir.cleanup)

    def read_ledger(self):
        return memorylab.load_jsonl(memorylab.LEDGER_PATH)

    def test_init_creates_expected_files(self):
        rc = memorylab.cmd_init(SimpleNamespace())
        self.assertEqual(rc, 0)
        self.assertTrue(memorylab.LEDGER_PATH.exists())
        self.assertTrue(memorylab.REGISTRY_PATH.exists())
        self.assertTrue(memorylab.RESULTS_TSV.exists())

    def test_log_archives_provenance_and_writes_registry(self):
        summary_source = self.root / "inputs" / "summary.json"
        log_source = self.root / "inputs" / "run.log"
        summary_source.parent.mkdir(parents=True, exist_ok=True)
        summary_source.write_text(json.dumps({
            "val_bpb": 0.991234,
            "peak_vram_mb": 4096.0,
            "training_seconds": 300.0,
        }) + "\n", encoding="utf-8")
        log_source.write_text("---\nval_bpb:          0.991234\npeak_vram_mb:     4096.0\n", encoding="utf-8")

        fixed_now = datetime(2026, 3, 11, 4, 30, tzinfo=timezone.utc)
        args = SimpleNamespace(
            description="increase matrix lr",
            status="keep",
            tags="optimizer,lr",
            mode="exploit",
            family="optimizer-sweep",
            hypothesis="higher matrix lr improves early progress",
            notes="first promising run",
            summary=str(summary_source),
            log=str(log_source),
            threshold=0.62,
            skip_archive=False,
            report=True,
            since_hours=16,
        )

        with mock.patch.object(memorylab, "get_git_context", return_value={
            "branch": "autoresearch/test",
            "commit": "abc1234",
            "parent_commit": "def5678",
        }), mock.patch.object(memorylab, "capture_git_text", return_value="diff --git a/train.py b/train.py\n+MATRIX_LR = 0.05\n"), mock.patch.object(memorylab, "utc_now", return_value=fixed_now):
            rc = memorylab.cmd_log(args)

        self.assertEqual(rc, 0)
        records = self.read_ledger()
        self.assertEqual(len(records), 1)
        record = records[0]
        self.assertEqual(record["family"], "optimizer-sweep")
        self.assertEqual(record["hypothesis"], "higher matrix lr improves early progress")
        self.assertIsNone(record["error"])
        self.assertEqual(record["novelty_guard"]["classification"], "novel")
        self.assertEqual(record["novelty_guard"]["mode"], "exploit")
        self.assertEqual(record["novelty_guard"]["policy"]["decision"], "caution")
        self.assertEqual(record["decision_packet"]["next_action"], "promote")
        self.assertEqual(record["decision_packet"]["priority"], "high")
        self.assertIn("artifacts", record)
        self.assertIn("archived_log_path", record["artifacts"])
        self.assertIn("archived_summary_path", record["artifacts"])
        self.assertIn("train_diff_path", record["artifacts"])
        self.assertIn("decision_packet_json_path", record["artifacts"])
        self.assertIn("decision_packet_md_path", record["artifacts"])

        archived_log = self.root / record["artifacts"]["archived_log_path"]
        archived_summary = self.root / record["artifacts"]["archived_summary_path"]
        archived_diff = self.root / record["artifacts"]["train_diff_path"]
        decision_json = self.root / record["artifacts"]["decision_packet_json_path"]
        decision_md = self.root / record["artifacts"]["decision_packet_md_path"]
        self.assertTrue(archived_log.exists())
        self.assertTrue(archived_summary.exists())
        self.assertTrue(archived_diff.exists())
        self.assertTrue(decision_json.exists())
        self.assertTrue(decision_md.exists())

        latest_report = memorylab.REPORTS_DIR / "latest.md"
        timestamped_report = memorylab.REPORTS_DIR / "20260311T043000Z.md"
        self.assertTrue(latest_report.exists())
        self.assertTrue(timestamped_report.exists())

        registry = json.loads(memorylab.REGISTRY_PATH.read_text(encoding="utf-8"))
        self.assertEqual(registry["champion"]["run_id"], record["run_id"])

    def test_check_returns_nonzero_for_similar_failure(self):
        memorylab.append_jsonl(memorylab.LEDGER_PATH, {
            "run_id": "run-1",
            "timestamp_utc": "2026-03-11T04:00:00Z",
            "branch": "autoresearch/test",
            "commit": "abc1234",
            "parent_commit": "base000",
            "family": "optimizer-sweep",
            "status": "discard",
            "description": "raise muon step size and reduce cooldown",
            "hypothesis": "",
            "tags": ["optimizer", "schedule"],
            "notes": "",
            "metrics": {"val_bpb": 1.002, "peak_vram_mb": 4096.0},
            "error": None,
            "artifacts": {},
            "novelty_guard": {"classification": "novel", "counts": {}, "match_count": 0, "top_matches": []},
        })

        args = SimpleNamespace(
            description="increase muon learning rate and shorten warmdown",
            tags="optimizer,schedule",
            family="",
            mode="exploit",
            threshold=0.5,
            limit=3,
            fail_on_similar=True,
        )
        rc = memorylab.cmd_check(args)
        self.assertEqual(rc, 2)
        history = memorylab.classify_history(
            self.read_ledger(),
            args.description,
            memorylab.normalize_tags(args.tags),
            family=args.family,
            threshold=args.threshold,
            limit=args.limit,
        )
        self.assertEqual(history["classification"], "repeat_failure")

    def test_check_known_success_is_history_aware_but_not_blocking(self):
        memorylab.append_jsonl(memorylab.LEDGER_PATH, {
            "run_id": "run-keep",
            "timestamp_utc": "2026-03-11T04:00:00Z",
            "branch": "autoresearch/test",
            "commit": "keep111",
            "parent_commit": "base000",
            "family": "optimizer-sweep",
            "status": "keep",
            "description": "increase matrix lr",
            "hypothesis": "",
            "tags": ["optimizer", "lr"],
            "notes": "",
            "metrics": {"val_bpb": 0.991, "peak_vram_mb": 4096.0},
            "error": None,
            "artifacts": {},
            "novelty_guard": {"classification": "novel", "counts": {}, "match_count": 0, "top_matches": []},
        })

        args = SimpleNamespace(
            description="raise matrix learning rate and reduce cooldown",
            tags="optimizer,schedule",
            family="optimizer-sweep",
            mode="exploit",
            threshold=0.5,
            limit=3,
            fail_on_similar=True,
        )
        rc = memorylab.cmd_check(args)
        self.assertEqual(rc, 0)
        history = memorylab.classify_history(
            self.read_ledger(),
            args.description,
            memorylab.normalize_tags(args.tags),
            family=args.family,
            threshold=args.threshold,
            limit=args.limit,
        )
        self.assertEqual(history["classification"], "incremental_followup")

    def test_explore_mode_blocks_incremental_followup(self):
        memorylab.append_jsonl(memorylab.LEDGER_PATH, {
            "run_id": "run-keep",
            "timestamp_utc": "2026-03-11T04:00:00Z",
            "branch": "autoresearch/test",
            "commit": "keep111",
            "parent_commit": "base000",
            "family": "optimizer-sweep",
            "status": "keep",
            "description": "increase matrix lr",
            "hypothesis": "",
            "tags": ["optimizer", "lr"],
            "notes": "",
            "metrics": {"val_bpb": 0.991, "peak_vram_mb": 4096.0},
            "error": None,
            "artifacts": {},
            "novelty_guard": {"classification": "novel", "counts": {}, "match_count": 0, "top_matches": []},
        })

        args = SimpleNamespace(
            description="raise matrix learning rate and reduce cooldown",
            tags="optimizer,schedule",
            family="optimizer-sweep",
            mode="explore",
            threshold=0.5,
            limit=3,
            fail_on_similar=True,
        )
        rc = memorylab.cmd_check(args)
        self.assertEqual(rc, 2)

    def test_replicate_mode_blocks_novel_idea(self):
        args = SimpleNamespace(
            description="invent a new attention schedule",
            tags="architecture,schedule",
            family="new-sweep",
            mode="replicate",
            threshold=0.5,
            limit=3,
            fail_on_similar=True,
        )
        rc = memorylab.cmd_check(args)
        self.assertEqual(rc, 2)

    def test_log_crash_uses_null_metrics_and_error_summary(self):
        log_source = self.root / "inputs" / "crash.log"
        log_source.parent.mkdir(parents=True, exist_ok=True)
        log_source.write_text(
            "step 00010\nTraceback (most recent call last):\nRuntimeError: CUDA out of memory\n",
            encoding="utf-8",
        )

        args = SimpleNamespace(
            description="double model width",
            status="crash",
            tags="architecture,oom",
            mode="exploit",
            family="width-sweep",
            hypothesis="larger width may improve model quality",
            notes="expected to be risky",
            summary=str(self.root / "stale-summary.json"),
            log=str(log_source),
            threshold=0.62,
            error="",
            skip_archive=True,
            report=False,
            since_hours=16,
        )

        with mock.patch.object(memorylab, "get_git_context", return_value={
            "branch": "autoresearch/test",
            "commit": "crsh123",
            "parent_commit": "base000",
        }):
            rc = memorylab.cmd_log(args)

        self.assertEqual(rc, 0)
        record = self.read_ledger()[0]
        self.assertIsNone(record["metrics"]["val_bpb"])
        self.assertIsNone(record["metrics"]["peak_vram_mb"])
        self.assertEqual(record["error"]["summary"], "RuntimeError: CUDA out of memory")
        self.assertEqual(record["decision_packet"]["next_action"], "fix_and_retry")
        self.assertEqual(record["decision_packet"]["hypothesis_status"], "inconclusive")
        results_lines = memorylab.RESULTS_TSV.read_text(encoding="utf-8").splitlines()
        self.assertIn("crsh123\t0.000000\t0.0\tcrash\tdouble model width", results_lines[-1])

    def test_log_kept_non_champion_novel_run_recommends_replicate(self):
        memorylab.append_jsonl(memorylab.LEDGER_PATH, {
            "run_id": "run-champion",
            "timestamp_utc": "2026-03-11T03:30:00Z",
            "branch": "autoresearch/test",
            "commit": "best111",
            "parent_commit": "base000",
            "family": "baseline",
            "status": "keep",
            "description": "baseline winner",
            "hypothesis": "",
            "tags": ["baseline"],
            "notes": "",
            "metrics": {"val_bpb": 0.991, "peak_vram_mb": 4096.0},
            "error": None,
            "artifacts": {},
            "novelty_guard": {"classification": "novel", "counts": {}, "match_count": 0, "top_matches": []},
        })

        summary_source = self.root / "inputs" / "novel-summary.json"
        log_source = self.root / "inputs" / "novel.log"
        summary_source.parent.mkdir(parents=True, exist_ok=True)
        summary_source.write_text(json.dumps({
            "val_bpb": 0.994,
            "peak_vram_mb": 4096.0,
        }) + "\n", encoding="utf-8")
        log_source.write_text("val_bpb:          0.994000\npeak_vram_mb:     4096.0\n", encoding="utf-8")

        args = SimpleNamespace(
            description="invent a new attention schedule",
            status="keep",
            tags="architecture,schedule",
            mode="exploit",
            family="new-sweep",
            hypothesis="a new schedule may improve token mixing",
            notes="novel but not obviously stronger",
            summary=str(summary_source),
            log=str(log_source),
            threshold=0.62,
            error="",
            skip_archive=True,
            report=False,
            since_hours=16,
        )

        with mock.patch.object(memorylab, "get_git_context", return_value={
            "branch": "autoresearch/test",
            "commit": "novl222",
            "parent_commit": "best111",
        }), mock.patch.object(memorylab, "capture_git_text", return_value=""), mock.patch.object(memorylab, "utc_now", return_value=datetime(2026, 3, 11, 5, 0, tzinfo=timezone.utc)):
            rc = memorylab.cmd_log(args)

        self.assertEqual(rc, 0)
        record = self.read_ledger()[-1]
        self.assertEqual(record["decision_packet"]["next_action"], "replicate")
        self.assertEqual(record["decision_packet"]["priority"], "medium")
        self.assertEqual(record["decision_packet"]["comparison"]["current_champion_run_id"], "run-champion")

    def test_report_generates_markdown_from_sample_ledger(self):
        sample_records = [
            {
                "run_id": "run-1",
                "timestamp_utc": "2026-03-11T04:00:00Z",
                "branch": "autoresearch/test",
                "commit": "aaa1111",
                "parent_commit": "base000",
                "family": "baseline",
                "status": "keep",
                "description": "baseline",
                "hypothesis": "",
                "tags": ["baseline"],
                "notes": "",
                "metrics": {"val_bpb": 0.999, "peak_vram_mb": 4096.0},
                "error": None,
                "artifacts": {},
                "novelty_guard": {"classification": "novel", "counts": {}, "match_count": 0, "top_matches": []},
                "decision_packet": {
                    "summary": "Promote run run-1: new champion at 0.999000 val_bpb.",
                    "next_action": "promote",
                    "priority": "high",
                },
            },
            {
                "run_id": "run-2",
                "timestamp_utc": "2026-03-11T05:00:00Z",
                "branch": "autoresearch/test",
                "commit": "bbb2222",
                "parent_commit": "aaa1111",
                "family": "optimizer-sweep",
                "status": "discard",
                "description": "increase matrix lr",
                "hypothesis": "",
                "tags": ["optimizer", "lr"],
                "notes": "",
                "metrics": {"val_bpb": 1.003, "peak_vram_mb": 4200.0},
                "error": None,
                "artifacts": {},
                "novelty_guard": {"classification": "novel", "counts": {}, "match_count": 0, "top_matches": []},
                "decision_packet": {
                    "summary": "Abandon run run-2: discarded follow-up in a weak line.",
                    "next_action": "abandon",
                    "priority": "low",
                },
            },
        ]
        for record in sample_records:
            memorylab.append_jsonl(memorylab.LEDGER_PATH, record)

        output_path = self.root / "report.md"
        rc = memorylab.cmd_report(SimpleNamespace(since_hours=24, output=str(output_path)))
        self.assertEqual(rc, 0)
        report_text = output_path.read_text(encoding="utf-8")
        self.assertIn("Morning Report", report_text)
        self.assertIn("Decision Queue", report_text)
        self.assertIn("Champion Board", report_text)
        self.assertIn("`promote`", report_text)
        self.assertIn("`aaa1111`", report_text)
        self.assertIn("`run-1`", report_text)

    def test_registry_is_run_centric_for_repeat_commit_runs(self):
        records = [
            {
                "run_id": "run-a",
                "timestamp_utc": "2026-03-11T04:00:00Z",
                "branch": "autoresearch/test",
                "commit": "same111",
                "parent_commit": "base000",
                "family": "seed-sweep",
                "status": "keep",
                "description": "repeatable winner",
                "hypothesis": "",
                "tags": ["seed"],
                "notes": "",
                "metrics": {"val_bpb": 0.991, "peak_vram_mb": 4096.0},
                "error": None,
                "artifacts": {},
                "novelty_guard": {"classification": "novel", "counts": {}, "match_count": 0, "top_matches": []},
            },
            {
                "run_id": "run-b",
                "timestamp_utc": "2026-03-11T04:10:00Z",
                "branch": "autoresearch/test",
                "commit": "same111",
                "parent_commit": "base000",
                "family": "seed-sweep",
                "status": "keep",
                "description": "repeatable second place",
                "hypothesis": "",
                "tags": ["seed"],
                "notes": "",
                "metrics": {"val_bpb": 0.992, "peak_vram_mb": 4096.0},
                "error": None,
                "artifacts": {},
                "novelty_guard": {"classification": "novel", "counts": {}, "match_count": 0, "top_matches": []},
            },
            {
                "run_id": "run-c",
                "timestamp_utc": "2026-03-11T04:20:00Z",
                "branch": "autoresearch/test",
                "commit": "diff222",
                "parent_commit": "same111",
                "family": "seed-sweep",
                "status": "keep",
                "description": "different commit",
                "hypothesis": "",
                "tags": ["seed"],
                "notes": "",
                "metrics": {"val_bpb": 0.995, "peak_vram_mb": 4096.0},
                "error": None,
                "artifacts": {},
                "novelty_guard": {"classification": "novel", "counts": {}, "match_count": 0, "top_matches": []},
            },
        ]
        registry = memorylab.build_registry(records)
        self.assertEqual(registry["champion"]["run_id"], "run-a")
        self.assertEqual(registry["challengers"][0]["run_id"], "run-b")


if __name__ == "__main__":
    unittest.main()
