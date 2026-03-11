"""Run registry and morning-report rendering for MemoryLab.

This module turns the raw ledger into the operator-facing views that make the
fork useful in practice:
- current champion and nearby challengers
- best lineages across commit ancestry
- repeated failure clusters
- recent runs and decision queue summaries
- the final morning report markdown

The registry is run-centric rather than commit-centric, which matters when the
same commit is rerun for replication or seed checks.
"""

from __future__ import annotations

from collections import Counter
from datetime import datetime, timedelta, timezone
from typing import Any

from .novelty import experiment_text, normalize_inline_text, similarity_score, tokenize


DECISION_PRIORITY = {
    "high": 0,
    "medium": 1,
    "low": 2,
}


def parse_timestamp(value: str) -> datetime:
    """Parse the UTC timestamp strings used throughout the MemoryLab ledger."""
    return datetime.fromisoformat(value.replace("Z", "+00:00"))


def metric_as_float(metrics: dict[str, Any], key: str) -> float | None:
    """Best-effort numeric conversion used by report builders."""
    value = metrics.get(key)
    if value in (None, ""):
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def format_metric(value: float | None, *, precision: int = 6, missing: str = "-") -> str:
    """Format optional numeric metrics for human-facing markdown tables."""
    if value is None:
        return missing
    return f"{value:.{precision}f}"


def display_run_id(record: dict[str, Any]) -> str:
    """Return the user-facing identifier for a run-centric record."""
    return record.get("run_id") or record.get("commit") or "-"


def format_memory_metric(metrics: dict[str, Any], missing: str = "-") -> str:
    """Format the stored peak VRAM metric as GB for report tables."""
    peak_vram_mb = metric_as_float(metrics, "peak_vram_mb")
    if peak_vram_mb is None:
        return missing
    return f"{peak_vram_mb / 1024:.1f}"


def memory_gb(metrics: dict[str, Any]) -> float:
    """Return peak memory in GB, matching the compatibility TSV display."""
    peak_vram_mb = metric_as_float(metrics, "peak_vram_mb") or 0.0
    return round(peak_vram_mb / 1024, 1)


def record_val_bpb(record: dict[str, Any]) -> float | None:
    """Extract the optimization metric from a stored run record."""
    return metric_as_float(record.get("metrics", {}), "val_bpb")


def derive_lineages(records: list[dict[str, Any]]) -> None:
    """Annotate each record with its lineage root based on git ancestry."""
    by_commit = {record.get("commit"): record for record in records if record.get("commit")}
    for record in records:
        current = record.get("commit")
        parent = record.get("parent_commit")
        root = current
        seen: set[str | None] = set()
        while parent and parent not in seen:
            seen.add(parent)
            if parent not in by_commit:
                root = parent
                break
            root = parent
            parent = by_commit[parent].get("parent_commit")
        record["lineage_root"] = root or current


def best_record(records: list[dict[str, Any]]) -> dict[str, Any] | None:
    """Choose the current best comparable run.

    Crashes are excluded. Kept runs are preferred over discarded ones, and lower
    `val_bpb` wins.
    """
    valid = [
        record for record in records
        if record.get("status") != "crash" and (record_val_bpb(record) or 0.0) > 0
    ]
    if not valid:
        return None
    kept = [record for record in valid if record.get("status") == "keep"]
    pool = kept or valid
    return min(pool, key=lambda record: ((record_val_bpb(record) or 999.0), record["timestamp_utc"]))


def build_failure_clusters(records: list[dict[str, Any]], *, limit: int = 5) -> list[dict[str, Any]]:
    """Group repeated bad ideas into operator-readable failure clusters."""
    failures = [record for record in records if record.get("status") in {"discard", "crash"}]
    clusters: list[dict[str, Any]] = []
    for record in failures:
        text = experiment_text(record)
        best_index = None
        best_score = 0.0
        for index, cluster in enumerate(clusters):
            score = similarity_score(text, cluster["prototype"])
            if score > best_score:
                best_index = index
                best_score = score
        # Clustering uses the same comparison text as the novelty guard so the
        # report highlights repeated classes of bad ideas instead of raw logs.
        if best_index is not None and best_score >= 0.6:
            clusters[best_index]["records"].append(record)
        else:
            clusters.append({"prototype": text, "records": [record]})

    summarized = []
    for cluster in clusters:
        records_in_cluster = cluster["records"]
        token_counts = Counter(
            token
            for record in records_in_cluster
            for token in tokenize(experiment_text(record))
        )
        label_tokens = [token for token, _ in token_counts.most_common(3)]
        label = ", ".join(label_tokens) if label_tokens else records_in_cluster[0]["description"]
        best_candidates = [
            record_val_bpb(record)
            for record in records_in_cluster
            if (record_val_bpb(record) or 0.0) > 0
        ]
        best_val = min(best_candidates) if best_candidates else None
        crash_count = sum(1 for record in records_in_cluster if record.get("status") == "crash")
        summarized.append({
            "label": label,
            "size": len(records_in_cluster),
            "crashes": crash_count,
            "best_val_bpb": best_val,
            "examples": [record["description"] for record in records_in_cluster[:3]],
        })
    summarized.sort(key=lambda row: (-row["size"], -row["crashes"], row["best_val_bpb"] or 999))
    return summarized[:limit]


def build_registry(records: list[dict[str, Any]]) -> dict[str, Any]:
    """Build the machine-readable champion/challenger registry."""
    derive_lineages(records)
    champion = best_record(records)
    valid = [
        record for record in records
        if record.get("status") != "crash" and (record_val_bpb(record) or 0.0) > 0
    ]
    valid.sort(key=lambda record: ((record_val_bpb(record) or 999.0), record["timestamp_utc"]))

    # Challengers are the next few best distinct runs after the champion. The
    # de-duplication key is run id, not commit, so repeated trials remain visible.
    challengers = []
    seen_runs = {display_run_id(champion)} if champion else set()
    for record in valid:
        run_id = display_run_id(record)
        if run_id in seen_runs:
            continue
        delta = None
        if champion:
            delta = (record_val_bpb(record) or 0.0) - (record_val_bpb(champion) or 0.0)
        challengers.append({
            "run_id": run_id,
            "commit": record["commit"],
            "lineage_root": record.get("lineage_root"),
            "family": record.get("family", ""),
            "status": record["status"],
            "val_bpb": record_val_bpb(record),
            "delta_to_champion": delta,
            "memory_gb": memory_gb(record["metrics"]),
            "description": record["description"],
        })
        seen_runs.add(run_id)
        if len(challengers) == 3:
            break

    lineage_map: dict[str, list[dict[str, Any]]] = {}
    for record in valid:
        lineage_map.setdefault(record.get("lineage_root") or record["commit"], []).append(record)
    best_lineages = []
    for lineage_root, lineage_records in lineage_map.items():
        lineage_records.sort(key=lambda record: ((record_val_bpb(record) or 999.0), record["timestamp_utc"]))
        leader = lineage_records[0]
        best_lineages.append({
            "lineage_root": lineage_root,
            "best_run_id": display_run_id(leader),
            "best_commit": leader["commit"],
            "best_val_bpb": record_val_bpb(leader),
            "num_runs": len(lineage_records),
            "num_unique_commits": len({record["commit"] for record in lineage_records}),
            "kept_runs": sum(1 for record in lineage_records if record["status"] == "keep"),
        })
    best_lineages.sort(key=lambda row: ((row["best_val_bpb"] or 999.0), -row["num_runs"]))

    status_counts = {status: 0 for status in ("crash", "discard", "keep")}
    for record in records:
        status_counts[record["status"]] += 1

    champion_payload = None
    if champion:
        champion_payload = {
            "run_id": display_run_id(champion),
            "commit": champion["commit"],
            "lineage_root": champion.get("lineage_root"),
            "family": champion.get("family", ""),
            "status": champion["status"],
            "val_bpb": record_val_bpb(champion),
            "memory_gb": memory_gb(champion["metrics"]),
            "description": champion["description"],
        }

    return {
        "updated_at": datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z"),
        "num_experiments": len(records),
        "status_counts": status_counts,
        "champion": champion_payload,
        "challengers": challengers,
        "best_lineages": best_lineages[:5],
        "failure_clusters": build_failure_clusters(records),
    }


def escape_markdown_cell(text: str) -> str:
    """Escape a small subset of markdown table delimiters for report safety."""
    return normalize_inline_text(text).replace("|", "\\|")


def render_board_rows(registry: dict[str, Any]) -> list[str]:
    """Render champion and challenger rows for the markdown report."""
    rows = ["| role | run_id | commit | val_bpb | delta | mem_gb | lineage | description |", "| --- | --- | --- | ---: | ---: | ---: | --- | --- |"]
    champion = registry.get("champion")
    if champion:
        rows.append(
            "| champion | `{run_id}` | `{commit}` | {val_bpb} | {delta} | {memory_gb:.1f} | `{lineage}` | {description} |".format(
                run_id=champion["run_id"],
                commit=champion["commit"],
                val_bpb=format_metric(champion["val_bpb"]),
                delta=format_metric(0.0),
                memory_gb=champion["memory_gb"],
                lineage=champion.get("lineage_root") or "-",
                description=escape_markdown_cell(champion["description"]),
            )
        )
    for challenger in registry.get("challengers", []):
        rows.append(
            "| challenger | `{run_id}` | `{commit}` | {val_bpb} | {delta} | {memory_gb:.1f} | `{lineage}` | {description} |".format(
                run_id=challenger["run_id"],
                commit=challenger["commit"],
                val_bpb=format_metric(challenger["val_bpb"]),
                delta=format_metric(challenger.get("delta_to_champion"), missing="-"),
                memory_gb=challenger["memory_gb"],
                lineage=challenger.get("lineage_root") or "-",
                description=escape_markdown_cell(challenger["description"]),
            )
        )
    if len(rows) == 2:
        rows.append("| - | - | - | - | - | - | - | No successful experiments recorded yet. |")
    return rows


def render_lineage_rows(registry: dict[str, Any]) -> list[str]:
    """Render the strongest lineages derived from commit ancestry."""
    rows = ["| lineage root | best run_id | best commit | best val_bpb | runs | unique commits | kept |", "| --- | --- | --- | ---: | ---: | ---: | ---: |"]
    lineages = registry.get("best_lineages", [])
    if not lineages:
        rows.append("| - | - | - | - | - | - | - |")
        return rows
    for lineage in lineages:
        rows.append(
            "| `{root}` | `{run_id}` | `{commit}` | {val_bpb} | {runs} | {unique_commits} | {kept} |".format(
                root=lineage["lineage_root"],
                run_id=lineage["best_run_id"],
                commit=lineage["best_commit"],
                val_bpb=format_metric(lineage["best_val_bpb"]),
                runs=lineage["num_runs"],
                unique_commits=lineage["num_unique_commits"],
                kept=lineage["kept_runs"],
            )
        )
    return rows


def render_recent_rows(records: list[dict[str, Any]], *, limit: int = 10) -> list[str]:
    """Render the most recent ledger entries for quick operator review."""
    rows = ["| time (UTC) | run_id | commit | status | val_bpb | mem_gb | description |", "| --- | --- | --- | --- | ---: | ---: | --- |"]
    recent = sorted(records, key=lambda record: record["timestamp_utc"], reverse=True)[:limit]
    if not recent:
        rows.append("| - | - | - | - | - | - | No experiments recorded yet. |")
        return rows
    for record in recent:
        metrics = record.get("metrics", {})
        rows.append(
            "| {time} | `{run_id}` | `{commit}` | {status} | {val_bpb} | {mem_gb} | {description} |".format(
                time=record["timestamp_utc"].replace("T", " "),
                run_id=display_run_id(record),
                commit=record["commit"],
                status=record["status"],
                val_bpb=format_metric(metric_as_float(metrics, "val_bpb")),
                mem_gb=format_memory_metric(metrics),
                description=escape_markdown_cell(record["description"]),
            )
        )
    return rows


def render_decision_rows(records: list[dict[str, Any]], *, limit: int = 5) -> list[str]:
    """Render the highest-priority recent decision packets.

    The sort order favors decision priority first and recency second so the top
    of the morning report surfaces what most likely needs human attention.
    """
    rows = ["| priority | next action | run_id | status | val_bpb | summary |", "| --- | --- | --- | --- | ---: | --- |"]
    decision_records = [record for record in records if record.get("decision_packet")]
    decision_records.sort(key=lambda record: parse_timestamp(record["timestamp_utc"]), reverse=True)
    decision_records.sort(key=lambda record: DECISION_PRIORITY.get(record["decision_packet"].get("priority"), 99))
    if not decision_records:
        rows.append("| - | - | - | - | - | No decision packets recorded in this window. |")
        return rows

    for record in decision_records[:limit]:
        packet = record["decision_packet"]
        rows.append(
            "| {priority} | `{action}` | `{run_id}` | {status} | {val_bpb} | {summary} |".format(
                priority=packet.get("priority", "-"),
                action=packet.get("next_action", "-"),
                run_id=display_run_id(record),
                status=record.get("status", "-"),
                val_bpb=format_metric(metric_as_float(record.get("metrics", {}), "val_bpb")),
                summary=escape_markdown_cell(packet.get("summary", "")),
            )
        )
    return rows


def generate_report(records: list[dict[str, Any]], *, since_hours: int) -> str:
    """Generate the human-readable morning report from the current ledger."""
    registry = build_registry(records)
    now = datetime.now(timezone.utc)
    cutoff = now - timedelta(hours=since_hours)
    overnight = [record for record in records if parse_timestamp(record["timestamp_utc"]) >= cutoff]
    overnight_counts = Counter(record["status"] for record in overnight)

    prior_records = [record for record in records if parse_timestamp(record["timestamp_utc"]) < cutoff]
    prior_champion = best_record(prior_records)
    current_champion = best_record(records)

    lines = [
        "# Morning Report",
        "",
        f"Generated: {now.replace(microsecond=0).isoformat()}",
        f"Window: last {since_hours} hours",
        "",
        "## Headline",
        f"- Experiments recorded overnight: {len(overnight)}",
        f"- Keep / discard / crash: {overnight_counts.get('keep', 0)} / {overnight_counts.get('discard', 0)} / {overnight_counts.get('crash', 0)}",
    ]

    if current_champion:
        lines.append(
            "- Current champion: run `{run_id}` on commit `{commit}` at {val_bpb}, {mem_gb:.1f} GB, {description}".format(
                run_id=display_run_id(current_champion),
                commit=current_champion["commit"],
                val_bpb=format_metric(record_val_bpb(current_champion)),
                mem_gb=memory_gb(current_champion["metrics"]),
                description=current_champion["description"],
            )
        )
    else:
        lines.append("- Current champion: none yet")

    if prior_champion and current_champion:
        delta = (record_val_bpb(current_champion) or 0.0) - (record_val_bpb(prior_champion) or 0.0)
        lines.append(f"- Change vs prior champion: {delta:+.6f} val_bpb")

    lines.extend([
        "",
        "## Decision Queue",
        *render_decision_rows(overnight),
        "",
        "## Champion Board",
        *render_board_rows(registry),
        "",
        "## Best Lineages",
        *render_lineage_rows(registry),
        "",
        "## Failure Clusters",
    ])

    if registry["failure_clusters"]:
        for cluster in registry["failure_clusters"]:
            example_text = "; ".join(cluster["examples"])
            lines.append(
                "- `{label}`: {size} runs, {crashes} crashes, best val_bpb {best}. Examples: {examples}".format(
                    label=cluster["label"],
                    size=cluster["size"],
                    crashes=cluster["crashes"],
                    best=format_metric(cluster["best_val_bpb"]),
                    examples=example_text,
                )
            )
    else:
        lines.append("- No failures logged yet.")

    lines.extend([
        "",
        "## Recent Ledger",
        *render_recent_rows(records),
    ])
    return "\n".join(lines) + "\n"
