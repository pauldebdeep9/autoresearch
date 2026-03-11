#!/usr/bin/env python3
"""
MemoryLab operator tooling for autoresearch.

This file is the operator-facing CLI added by the fork. Upstream autoresearch
keeps the research loop intentionally minimal; MemoryLab adds the surrounding
research-operations layer:

- structured experiment ledger
- history-aware novelty guard across prior failures and successes
- run-centric champion/challenger registry
- decision packets with next-action recommendations
- archived provenance for each run
- human-readable morning report

The CLI is intentionally thin. Most reasoning lives in `memorylab_core/`, while
this module owns:
- file layout and persistence
- parsing summaries and logs
- git provenance capture
- command-line wiring
"""

from __future__ import annotations

import argparse
import json
import re
import shutil
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from memorylab_core import decisions as decision_core
from memorylab_core import novelty as novelty_core
from memorylab_core import registry as registry_core


ROOT = Path(__file__).resolve().parent
RESULTS_TSV = ROOT / "results.tsv"
MEMORYLAB_DIR = ROOT / "results" / "memorylab"
LEDGER_PATH = MEMORYLAB_DIR / "experiments.jsonl"
REGISTRY_PATH = MEMORYLAB_DIR / "champion_challenger.json"
REPORTS_DIR = MEMORYLAB_DIR / "reports"
RUNS_DIR = MEMORYLAB_DIR / "runs"

VALID_STATUSES = {"keep", "discard", "crash"}
INT_METRICS = {"num_steps", "depth"}
SUMMARY_RE = re.compile(r"^([a-z_]+):\s+(.+?)\s*$")


def utc_now() -> datetime:
    """Return the current wall-clock time in UTC."""
    return datetime.now(timezone.utc)


def timestamp_utc() -> str:
    """Return a stable ISO-like UTC timestamp for run records."""
    return utc_now().replace(microsecond=0).isoformat().replace("+00:00", "Z")


def parse_timestamp(value: str) -> datetime:
    """Parse the timestamp format used in the MemoryLab ledger."""
    return datetime.fromisoformat(value.replace("Z", "+00:00"))


def ensure_store() -> None:
    """Create the on-disk MemoryLab directory layout if it does not exist yet."""
    MEMORYLAB_DIR.mkdir(parents=True, exist_ok=True)
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    RUNS_DIR.mkdir(parents=True, exist_ok=True)


def init_results_tsv() -> None:
    """Initialize the compatibility TSV used by upstream-style workflows."""
    if RESULTS_TSV.exists():
        return
    RESULTS_TSV.write_text("commit\tval_bpb\tmemory_gb\tstatus\tdescription\n", encoding="utf-8")


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    """Load newline-delimited JSON records from disk."""
    if not path.exists():
        return []
    rows = []
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        rows.append(json.loads(line))
    return rows


def append_jsonl(path: Path, payload: dict[str, Any]) -> None:
    """Append one JSON object to a JSONL file."""
    ensure_store()
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(payload, sort_keys=True) + "\n")


def write_json(path: Path, payload: dict[str, Any]) -> None:
    """Write a stable pretty-printed JSON file."""
    ensure_store()
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def git_output(args: list[str], *, allow_empty: bool = False) -> str | None:
    """Run a git command and return stripped stdout.

    MemoryLab records git context as provenance so later readers can connect a
    run back to the exact training code that produced it.
    """
    result = subprocess.run(
        ["git", *args],
        cwd=ROOT,
        capture_output=True,
        text=True,
        check=False,
    )
    if result.returncode == 0:
        value = result.stdout.strip()
        return value or None
    if allow_empty:
        return None
    raise RuntimeError(result.stderr.strip() or f"git {' '.join(args)} failed")


def capture_git_text(args: list[str]) -> str:
    """Capture free-form git output such as a `train.py` diff."""
    result = subprocess.run(
        ["git", *args],
        cwd=ROOT,
        capture_output=True,
        text=True,
        check=False,
    )
    if result.returncode == 0:
        return result.stdout
    return ""


def get_git_context() -> dict[str, str | None]:
    """Return the branch, commit, and parent commit for the current workspace."""
    return {
        "branch": git_output(["rev-parse", "--abbrev-ref", "HEAD"], allow_empty=True),
        "commit": git_output(["rev-parse", "--short=7", "HEAD"], allow_empty=True),
        "parent_commit": git_output(["rev-parse", "--short=7", "HEAD^"], allow_empty=True),
    }


# The next block intentionally re-exports core helpers through the CLI module.
# Tests and external callers already reach some of these via `memorylab.py`, so
# keeping these wrappers small avoids pushing callers into implementation files.
def normalize_tags(raw_tags: str | None) -> list[str]:
    return novelty_core.normalize_tags(raw_tags)


def normalize_text(text: str) -> str:
    return novelty_core.normalize_text(text)


def normalize_inline_text(text: str) -> str:
    return novelty_core.normalize_inline_text(text)


def escape_markdown_cell(text: str) -> str:
    return registry_core.escape_markdown_cell(text)


def path_label(path: Path | None) -> str:
    if path is None:
        return ""
    resolved = path.resolve()
    try:
        return str(resolved.relative_to(ROOT))
    except ValueError:
        return str(resolved)


def timestamp_slug(value: datetime | None = None) -> str:
    current = value or utc_now()
    return current.strftime("%Y%m%dT%H%M%SZ")


def metric_as_float(metrics: dict[str, Any], key: str) -> float | None:
    return registry_core.metric_as_float(metrics, key)


def format_metric(value: float | None, *, precision: int = 6, missing: str = "-") -> str:
    return registry_core.format_metric(value, precision=precision, missing=missing)


def display_run_id(record: dict[str, Any]) -> str:
    return registry_core.display_run_id(record)


def format_memory_metric(metrics: dict[str, Any], missing: str = "-") -> str:
    return registry_core.format_memory_metric(metrics, missing=missing)


def tokenize(text: str) -> list[str]:
    return novelty_core.tokenize(text)


def experiment_text(record: dict[str, Any]) -> str:
    return novelty_core.experiment_text(record)


def similarity_score(a: str, b: str) -> float:
    return novelty_core.similarity_score(a, b)


def find_similar_failures(
    records: list[dict[str, Any]],
    description: str,
    tags: list[str],
    *,
    threshold: float,
    limit: int = 3,
) -> list[dict[str, Any]]:
    return novelty_core.find_similar_failures(
        records,
        description,
        tags,
        threshold=threshold,
        limit=limit,
    )


def classify_history(
    records: list[dict[str, Any]],
    description: str,
    tags: list[str],
    *,
    family: str = "",
    hypothesis: str = "",
    threshold: float,
    limit: int = 5,
) -> dict[str, Any]:
    return novelty_core.classify_history(
        records,
        description,
        tags,
        family=family,
        hypothesis=hypothesis,
        threshold=threshold,
        limit=limit,
    )


def normalize_mode(mode: str | None) -> str:
    return novelty_core.normalize_mode(mode)


def effective_threshold(threshold: float, mode: str) -> float:
    return novelty_core.effective_threshold(threshold, mode)


def evaluate_policy(classification: str, mode: str) -> dict[str, str]:
    return novelty_core.evaluate_policy(classification, mode)


def build_decision_packet(
    record: dict[str, Any],
    *,
    prior_champion: dict[str, Any] | None,
    current_champion: dict[str, Any] | None,
) -> dict[str, Any]:
    """Build the structured decision packet archived with each run."""
    return decision_core.build_decision_packet(
        record,
        prior_champion=prior_champion,
        current_champion=current_champion,
    )


def parse_metric_value(key: str, value: str) -> int | float | str:
    """Parse summary lines, preserving integer semantics for selected fields."""
    if key in INT_METRICS:
        return int(float(value))
    try:
        return float(value)
    except ValueError:
        return value.strip()


def parse_summary_text(text: str) -> dict[str, Any]:
    """Parse the human-readable metric block emitted by `train.py`."""
    metrics: dict[str, Any] = {}
    for line in text.splitlines():
        match = SUMMARY_RE.match(line.strip())
        if not match:
            continue
        key, raw_value = match.groups()
        metrics[key] = parse_metric_value(key, raw_value)
    return metrics


def extract_error_info(log_path: Path | None, explicit_error: str = "") -> dict[str, Any]:
    """Build a structured crash summary from a log file or explicit override."""
    if explicit_error:
        return {
            "summary": normalize_inline_text(explicit_error),
            "tail": [],
            "source": "cli",
        }

    if not log_path or not log_path.exists():
        return {
            "summary": "Run crashed without a captured log.",
            "tail": [],
            "source": "missing-log",
        }

    lines = [line.rstrip() for line in log_path.read_text(encoding="utf-8", errors="replace").splitlines()]
    nonempty = [normalize_inline_text(line) for line in lines if line.strip()]
    tail = nonempty[-8:]
    traceback_index = next((index for index, line in enumerate(lines) if line.startswith("Traceback")), None)

    summary = ""
    if traceback_index is not None:
        for line in reversed(lines[traceback_index:]):
            if line.strip():
                summary = normalize_inline_text(line)
                break
    if not summary and nonempty:
        summary = nonempty[-1]
    if not summary:
        summary = "Run crashed; inspect run.log for details."

    return {
        "summary": summary,
        "tail": tail,
        "source": "run.log",
    }


def parse_metrics(summary_path: Path | None, log_path: Path | None, status: str) -> dict[str, Any]:
    """Load structured metrics for a run.

    Preferred source is the JSON summary sidecar written by `train.py` when
    `AUTORESEARCH_SUMMARY_PATH` is set. Parsing the run log remains as a fallback
    so the CLI stays usable with older or more manual workflows.
    """
    metrics: dict[str, Any] = {}
    if status != "crash" and summary_path and summary_path.exists():
        metrics.update(json.loads(summary_path.read_text(encoding="utf-8")))
    if (not metrics or "val_bpb" not in metrics) and log_path and log_path.exists():
        metrics.update(parse_summary_text(log_path.read_text(encoding="utf-8")))
    if status == "crash":
        metrics["val_bpb"] = None
        metrics["peak_vram_mb"] = metric_as_float(metrics, "peak_vram_mb")
        return metrics
    if "val_bpb" not in metrics or "peak_vram_mb" not in metrics:
        raise ValueError("Could not find val_bpb/peak_vram_mb in summary or run log.")
    return metrics


def memory_gb(metrics: dict[str, Any]) -> float:
    return registry_core.memory_gb(metrics)


def append_results_row(record: dict[str, Any]) -> None:
    """Append the compatibility TSV row expected by the upstream workflow."""
    init_results_tsv()
    metrics = record["metrics"]
    val_bpb = metric_as_float(metrics, "val_bpb") or 0.0
    line = (
        f"{record['commit']}\t"
        f"{val_bpb:.6f}\t"
        f"{memory_gb(metrics):.1f}\t"
        f"{record['status']}\t"
        f"{record['description']}\n"
    )
    with RESULTS_TSV.open("a", encoding="utf-8") as handle:
        handle.write(line)


def archive_run_artifacts(
    *,
    run_id: str,
    commit: str,
    parent_commit: str | None,
    log_path: Path | None,
    summary_path: Path | None,
    archive: bool,
) -> dict[str, str]:
    """Archive the raw artifacts needed to understand or replay a run.

    This is one of the fork's biggest practical additions: instead of keeping
    only a metric line, the repo retains the raw log, structured summary, and
    `train.py` diff that explain what actually happened.
    """
    artifacts: dict[str, str] = {}
    if log_path is not None:
        artifacts["source_log_path"] = path_label(log_path)
    if summary_path is not None:
        artifacts["source_summary_path"] = path_label(summary_path)
    if not archive:
        return artifacts

    run_dir = RUNS_DIR / run_id
    run_dir.mkdir(parents=True, exist_ok=True)
    artifacts["run_dir"] = path_label(run_dir)

    if log_path and log_path.exists():
        archived_log_path = run_dir / "run.log"
        shutil.copy2(log_path, archived_log_path)
        artifacts["archived_log_path"] = path_label(archived_log_path)

    if summary_path and summary_path.exists():
        archived_summary_path = run_dir / "summary.json"
        shutil.copy2(summary_path, archived_summary_path)
        artifacts["archived_summary_path"] = path_label(archived_summary_path)

    # Prefer the exact diff from parent to current commit when available. This
    # keeps the archived patch aligned with the experimental step that produced
    # the run, not just the current working tree.
    diff_text = ""
    if parent_commit:
        diff_text = capture_git_text(["diff", parent_commit, commit, "--", "train.py"])
    elif commit:
        diff_text = capture_git_text(["show", commit, "--", "train.py"])
    if diff_text:
        diff_path = run_dir / "train.patch"
        diff_path.write_text(diff_text, encoding="utf-8")
        artifacts["train_diff_path"] = path_label(diff_path)

    return artifacts


def derive_lineages(records: list[dict[str, Any]]) -> None:
    registry_core.derive_lineages(records)


def record_val_bpb(record: dict[str, Any]) -> float | None:
    return registry_core.record_val_bpb(record)


def best_record(records: list[dict[str, Any]]) -> dict[str, Any] | None:
    return registry_core.best_record(records)


def build_failure_clusters(records: list[dict[str, Any]], *, limit: int = 5) -> list[dict[str, Any]]:
    return registry_core.build_failure_clusters(records, limit=limit)


def build_registry(records: list[dict[str, Any]]) -> dict[str, Any]:
    return registry_core.build_registry(records)


def render_board_rows(registry: dict[str, Any]) -> list[str]:
    return registry_core.render_board_rows(registry)


def render_lineage_rows(registry: dict[str, Any]) -> list[str]:
    return registry_core.render_lineage_rows(registry)


def render_recent_rows(records: list[dict[str, Any]], *, limit: int = 10) -> list[str]:
    return registry_core.render_recent_rows(records, limit=limit)


def generate_report(records: list[dict[str, Any]], *, since_hours: int) -> str:
    return registry_core.generate_report(records, since_hours=since_hours)


def archive_decision_packet(run_dir: Path, packet: dict[str, Any]) -> dict[str, str]:
    """Write both machine-readable and skim-friendly decision packet artifacts."""
    decision_json_path = run_dir / "decision_packet.json"
    decision_md_path = run_dir / "decision_packet.md"
    decision_json_path.write_text(json.dumps(packet, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    decision_md_path.write_text(decision_core.render_decision_packet_markdown(packet), encoding="utf-8")
    return {
        "decision_packet_json_path": path_label(decision_json_path),
        "decision_packet_md_path": path_label(decision_md_path),
    }


def cmd_init(_: argparse.Namespace) -> int:
    """Initialize the MemoryLab storage layout on disk."""
    ensure_store()
    init_results_tsv()
    if not LEDGER_PATH.exists():
        LEDGER_PATH.write_text("", encoding="utf-8")
    if not REGISTRY_PATH.exists():
        write_json(REGISTRY_PATH, build_registry([]))
    print(f"Initialized MemoryLab in {MEMORYLAB_DIR}")
    print(f"Ledger: {LEDGER_PATH}")
    print(f"Registry: {REGISTRY_PATH}")
    print(f"Results TSV: {RESULTS_TSV}")
    return 0


def cmd_check(args: argparse.Namespace) -> int:
    """Run the history-aware novelty guard for a proposed future experiment."""
    records = load_jsonl(LEDGER_PATH)
    tags = normalize_tags(args.tags)
    family = normalize_inline_text(args.family) if getattr(args, "family", "") else ""
    mode = normalize_mode(getattr(args, "mode", None))
    threshold = effective_threshold(args.threshold, mode)
    history = classify_history(
        records,
        args.description,
        tags,
        family=family,
        threshold=threshold,
        limit=args.limit,
    )
    classification = history["classification"]
    policy = evaluate_policy(classification, mode)
    matches = history["top_matches"]
    counts = history["counts"]
    print(
        "Novelty guard [{mode}]: {classification} -> {decision} "
        "({rationale}; threshold={threshold:.2f}) "
        "(duplicate={duplicate_run}, repeat_failure={repeat_failure}, "
        "incremental_followup={incremental_followup}, known_success={known_success}).".format(
            mode=mode,
            classification=classification,
            decision=policy["decision"],
            rationale=policy["rationale"],
            threshold=threshold,
            duplicate_run=counts["duplicate_run"],
            repeat_failure=counts["repeat_failure"],
            incremental_followup=counts["incremental_followup"],
            known_success=counts["known_success"],
        )
    )
    if matches:
        for match in matches:
            print(
                "- category={category} score={score:.3f} run={run_id} commit={commit} status={status} val_bpb={val_bpb} desc={description}".format(
                    category=match["category"],
                    score=match["score"],
                    run_id=match.get("run_id") or "-",
                    commit=match["commit"],
                    status=match["status"],
                    val_bpb=format_metric(match["val_bpb"]),
                    description=match["description"],
                )
            )
    else:
        print("- No close prior runs matched at this threshold.")
    return 2 if args.fail_on_similar and policy["decision"] == "block" else 0


def cmd_log(args: argparse.Namespace) -> int:
    """Log a completed run, update the registry, and optionally refresh reports."""
    if args.status not in VALID_STATUSES:
        raise ValueError(f"Status must be one of: {', '.join(sorted(VALID_STATUSES))}")

    ensure_store()
    init_results_tsv()

    records = load_jsonl(LEDGER_PATH)
    tags = normalize_tags(args.tags)
    family = normalize_inline_text(args.family) if args.family else ""
    hypothesis = normalize_inline_text(args.hypothesis) if args.hypothesis else ""
    mode = normalize_mode(getattr(args, "mode", None))
    threshold = effective_threshold(args.threshold, mode)
    novelty_history = classify_history(
        records,
        args.description,
        tags,
        family=family,
        hypothesis=hypothesis,
        threshold=threshold,
        limit=5,
    )
    novelty_policy = evaluate_policy(novelty_history["classification"], mode)
    summary_path = Path(args.summary).resolve() if args.summary else None
    log_path = Path(args.log).resolve() if args.log else None
    metrics = parse_metrics(summary_path, log_path, args.status)
    error_info = extract_error_info(log_path, explicit_error=args.error) if args.status == "crash" else None

    git_context = get_git_context()
    if not git_context["commit"]:
        raise RuntimeError("Could not determine current git commit.")

    recorded_at = timestamp_utc()
    run_id = f"{recorded_at.replace(':', '').replace('-', '')}-{git_context['commit']}"
    artifacts = archive_run_artifacts(
        run_id=run_id,
        commit=git_context["commit"],
        parent_commit=git_context["parent_commit"],
        log_path=log_path,
        summary_path=summary_path,
        archive=not args.skip_archive,
    )
    record = {
        "run_id": run_id,
        "timestamp_utc": recorded_at,
        "branch": git_context["branch"],
        "commit": git_context["commit"],
        "parent_commit": git_context["parent_commit"],
        "family": family,
        "status": args.status,
        "description": normalize_inline_text(args.description),
        "hypothesis": hypothesis,
        "tags": tags,
        "notes": normalize_inline_text(args.notes) if args.notes else "",
        "metrics": metrics,
        "error": error_info,
        "artifacts": artifacts,
        "novelty_guard": {
            **novelty_history,
            "mode": mode,
            "effective_threshold": threshold,
            "policy": novelty_policy,
        },
    }

    # Decision packets compare the new run both to the old champion and to the
    # registry after inserting the current run. This captures whether the run
    # actually moved the frontier or only produced a useful follow-up signal.
    prior_champion = best_record(records)
    records_with_current = [*records, record]
    registry = build_registry(records_with_current)
    decision_packet = build_decision_packet(
        record,
        prior_champion=prior_champion,
        current_champion=registry.get("champion"),
    )
    record["decision_packet"] = decision_packet

    if not args.skip_archive and artifacts.get("run_dir"):
        run_dir = Path(artifacts["run_dir"])
        if not run_dir.is_absolute():
            run_dir = ROOT / run_dir
        record["artifacts"].update(archive_decision_packet(run_dir, decision_packet))

    append_jsonl(LEDGER_PATH, record)
    append_results_row(record)
    write_json(REGISTRY_PATH, registry)

    if args.report:
        report_text = generate_report(records_with_current, since_hours=args.since_hours)
        latest_path = REPORTS_DIR / "latest.md"
        dated_path = REPORTS_DIR / f"{timestamp_slug()}.md"
        latest_path.write_text(report_text, encoding="utf-8")
        dated_path.write_text(report_text, encoding="utf-8")
        print(f"Wrote report to {latest_path}")
        print(f"Wrote report to {dated_path}")

    print(
        "Logged run={run_id} commit={commit} status={status} val_bpb={val_bpb} mem_gb={mem_gb}".format(
            run_id=record["run_id"],
            commit=record["commit"],
            status=record["status"],
            val_bpb=format_metric(metric_as_float(record["metrics"], "val_bpb")),
            mem_gb=format_memory_metric(record["metrics"]),
        )
    )
    if error_info:
        print(f"Crash summary: {error_info['summary']}")
    if novelty_history["classification"] != "novel":
        print(
            "Novelty guard: mode={mode} classification={classification} decision={decision}".format(
                mode=mode,
                classification=novelty_history["classification"],
                decision=novelty_policy["decision"],
            )
        )
    print(
        "Decision packet: action={action} priority={priority} summary={summary}".format(
            action=decision_packet["next_action"],
            priority=decision_packet["priority"],
            summary=decision_packet["summary"],
        )
    )
    champion = registry.get("champion")
    if champion:
        print(
            "Champion: {run_id} {commit} {val_bpb} {description}".format(
                run_id=champion["run_id"],
                commit=champion["commit"],
                val_bpb=format_metric(champion["val_bpb"]),
                description=champion["description"],
            )
        )
    return 0


def cmd_report(args: argparse.Namespace) -> int:
    """Render the current morning report from the stored ledger."""
    records = load_jsonl(LEDGER_PATH)
    report_text = generate_report(records, since_hours=args.since_hours)
    ensure_store()
    if args.output:
        output_path = Path(args.output).resolve()
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(report_text, encoding="utf-8")
        print(f"Wrote report to {output_path}")
        return 0

    latest_path = REPORTS_DIR / "latest.md"
    dated_path = REPORTS_DIR / f"{timestamp_slug()}.md"
    latest_path.write_text(report_text, encoding="utf-8")
    dated_path.write_text(report_text, encoding="utf-8")
    print(f"Wrote report to {latest_path}")
    print(f"Wrote report to {dated_path}")
    return 0


def build_parser() -> argparse.ArgumentParser:
    """Construct the CLI parser for the MemoryLab command surface."""
    parser = argparse.ArgumentParser(description="MemoryLab workflow tools for autoresearch.")
    subparsers = parser.add_subparsers(dest="command", required=True)

    init_parser = subparsers.add_parser("init", help="Initialize the MemoryLab ledger and registry.")
    init_parser.set_defaults(func=cmd_init)

    check_parser = subparsers.add_parser("check", help="Run the history-aware novelty guard against prior runs.")
    check_parser.add_argument("--description", required=True, help="One-line description of the planned experiment.")
    check_parser.add_argument("--tags", default="", help="Comma-separated tags to improve matching.")
    check_parser.add_argument("--family", default="", help="Optional experiment family to distinguish follow-up work from novelty.")
    check_parser.add_argument("--mode", default="exploit", choices=["explore", "exploit", "replicate"], help="Novelty policy mode.")
    check_parser.add_argument("--threshold", type=float, default=0.62, help="Base similarity threshold before mode-specific adjustment.")
    check_parser.add_argument("--limit", type=int, default=3, help="Maximum number of similar prior runs to show.")
    check_parser.add_argument("--fail-on-similar", action="store_true", help="Exit non-zero if the selected novelty policy blocks the idea.")
    check_parser.set_defaults(func=cmd_check)

    log_parser = subparsers.add_parser("log", help="Log a completed experiment into the MemoryLab ledger.")
    log_parser.add_argument("--description", required=True, help="One-line description of what the experiment tried.")
    log_parser.add_argument("--status", required=True, choices=sorted(VALID_STATUSES), help="Outcome of the experiment.")
    log_parser.add_argument("--tags", default="", help="Comma-separated tags for clustering and novelty checks.")
    log_parser.add_argument("--mode", default="exploit", choices=["explore", "exploit", "replicate"], help="Novelty policy mode recorded with the run.")
    log_parser.add_argument("--family", default="", help="Optional experiment family or sweep label.")
    log_parser.add_argument("--hypothesis", default="", help="Optional one-line hypothesis captured alongside the run.")
    log_parser.add_argument("--notes", default="", help="Optional extra notes for the ledger entry.")
    log_parser.add_argument("--log", default="", help="Path to run.log for summary parsing fallback.")
    log_parser.add_argument("--summary", default="", help="Path to structured summary JSON emitted by train.py.")
    log_parser.add_argument("--threshold", type=float, default=0.62, help="Base novelty threshold to store alongside the run.")
    log_parser.add_argument("--error", default="", help="Optional crash summary override; used when --status crash.")
    log_parser.add_argument("--skip-archive", action="store_true", help="Do not copy log/summary/diff artifacts into results/memorylab/runs/.")
    log_parser.add_argument("--report", action="store_true", help="Refresh the morning report after logging.")
    log_parser.add_argument("--since-hours", type=int, default=16, help="Window size for the generated report.")
    log_parser.set_defaults(func=cmd_log)

    report_parser = subparsers.add_parser("report", help="Generate a human-readable morning report.")
    report_parser.add_argument("--since-hours", type=int, default=16, help="Window size for the report.")
    report_parser.add_argument("--output", default="", help="Optional path for a single output file.")
    report_parser.set_defaults(func=cmd_report)

    return parser


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()
    try:
        return args.func(args)
    except Exception as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
