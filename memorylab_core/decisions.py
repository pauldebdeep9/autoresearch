"""Decision-packet synthesis for MemoryLab.

Decision packets are a fork-specific layer that turns raw run history into an
operator action. Upstream autoresearch tells you what `val_bpb` a run reached;
this module tries to answer what the operator should do next.

The packet combines:
- run outcome and metrics
- novelty classification and policy decision
- comparison to the prior/current champion
- a next action such as `promote`, `replicate`, or `fix_and_retry`

The logic is intentionally simple and inspectable. It is meant to be stable
lab infrastructure, not a hidden model.
"""

from __future__ import annotations

from typing import Any

from .novelty import normalize_inline_text


PROMISING_DELTA = 0.0015
TOP_MATCH_KEYS = ("run_id", "commit", "category", "status", "score", "family", "description", "val_bpb")


def metric_as_float(metrics: dict[str, Any], key: str) -> float | None:
    """Best-effort numeric conversion for packet rendering and comparisons."""
    value = metrics.get(key)
    if value in (None, ""):
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def record_val_bpb(record: dict[str, Any]) -> float | None:
    """Extract the main optimization metric from a run record."""
    return metric_as_float(record.get("metrics", {}), "val_bpb")


def memory_gb(record: dict[str, Any]) -> float | None:
    """Convert peak VRAM in MB into the human-facing GB display used in reports."""
    peak_vram_mb = metric_as_float(record.get("metrics", {}), "peak_vram_mb")
    if peak_vram_mb is None:
        return None
    return round(peak_vram_mb / 1024, 1)


def display_run_id(record: dict[str, Any] | None) -> str | None:
    """Return the canonical run identifier, falling back to commit when needed."""
    if not record:
        return None
    return record.get("run_id") or record.get("commit")


def top_match(novelty_guard: dict[str, Any]) -> dict[str, Any] | None:
    """Trim novelty evidence down to the fields that are useful in a packet."""
    matches = novelty_guard.get("top_matches") or []
    if not matches:
        return None
    match = matches[0]
    return {key: match.get(key) for key in TOP_MATCH_KEYS}


def is_memory_error(summary: str) -> bool:
    """Detect crash summaries that likely indicate capacity pressure."""
    normalized = normalize_inline_text(summary).lower()
    return any(token in normalized for token in ("oom", "out of memory", "cuda out of memory", "vram", "memory"))


def format_delta(delta: float | None) -> str:
    """Format champion deltas for packet summaries."""
    if delta is None:
        return "-"
    return f"{delta:+.6f}"


def classify_hypothesis(record: dict[str, Any], *, is_new_champion: bool, delta_to_current: float | None) -> str:
    """Estimate whether the run supported the recorded hypothesis.

    This is intentionally coarse. It exists to make the packet readable at a
    glance, not to serve as a formal statistical conclusion.
    """
    hypothesis = normalize_inline_text(record.get("hypothesis", ""))
    if not hypothesis:
        return "not_stated"
    status = record.get("status")
    if status == "crash":
        return "inconclusive"
    if status == "keep" and is_new_champion:
        return "supported"
    if status == "keep":
        return "mixed"
    if delta_to_current is not None and delta_to_current <= PROMISING_DELTA:
        return "mixed"
    return "unsupported"


def choose_action(
    record: dict[str, Any],
    *,
    novelty_guard: dict[str, Any],
    is_new_champion: bool,
    delta_to_current: float | None,
) -> tuple[str, str, str]:
    """Choose the next operator action from outcome, novelty, and champion context."""
    status = record.get("status")
    classification = novelty_guard.get("classification", "novel")
    error_summary = record.get("error", {}).get("summary", "") if record.get("error") else ""

    # Crash decisions are always highest priority because there is no usable
    # metric until the operator decides whether the failure is fixable.
    if status == "crash":
        if is_memory_error(error_summary):
            return "fix_and_retry", "high", "The run crashed on a memory-related error before producing a usable result."
        return "investigate_crash", "high", "The run crashed before producing a usable result."

    if status == "keep" and is_new_champion:
        return "promote", "high", "This run is the new champion and should become the default branch point."

    # Successful non-champion runs still matter. The packet distinguishes
    # "promote now" from "confirm" and "follow this branch a bit further".
    if status == "keep":
        if classification == "novel":
            return "replicate", "medium", "This kept run is novel enough that it should be confirmed before branching harder."
        if classification == "duplicate_run":
            return "replicate", "medium", "This kept run closely matches prior work and is best treated as a repeatability check."
        if delta_to_current is not None and delta_to_current <= PROMISING_DELTA:
            return "branch_followup", "high", "This kept run stayed close to the champion in a promising line of work."
        return "branch_followup", "medium", "This kept run extends a known-good line and is a reasonable branch point."

    if classification in {"repeat_failure", "duplicate_run"}:
        return "abandon", "low", "This discarded run did not add enough signal beyond already-known bad territory."
    if delta_to_current is not None and delta_to_current <= PROMISING_DELTA:
        return "retry", "medium", "This discarded run landed close to the champion and may be worth tightening or rerunning."
    if classification in {"incremental_followup", "known_success"}:
        return "branch_followup", "medium", "The exact run was discarded, but the surrounding family still looks worth pursuing."
    return "abandon", "low", "This discarded run does not justify another cycle."


def build_summary(
    record: dict[str, Any],
    *,
    next_action: str,
    is_new_champion: bool,
    delta_to_prior: float | None,
    delta_to_current: float | None,
) -> str:
    """Render a one-line human summary for the packet and report."""
    run_id = display_run_id(record) or "-"
    val_bpb = record_val_bpb(record)
    metrics_text = f"{val_bpb:.6f} val_bpb" if val_bpb is not None else "no usable metrics"
    if next_action == "promote":
        if delta_to_prior is None:
            return f"Promote run {run_id}: first champion recorded at {metrics_text}."
        return f"Promote run {run_id}: new champion at {metrics_text} ({format_delta(delta_to_prior)} vs prior champion)."
    if next_action == "replicate":
        return f"Replicate run {run_id}: kept result at {metrics_text} to confirm the signal."
    if next_action == "branch_followup":
        return f"Branch from run {run_id}: {metrics_text}, {format_delta(delta_to_current)} vs current champion."
    if next_action == "retry":
        return f"Retry around run {run_id}: discarded result finished {format_delta(delta_to_current)} from the current champion."
    if next_action == "fix_and_retry":
        return f"Fix and retry run {run_id}: crash prevented a usable measurement."
    if next_action == "investigate_crash":
        return f"Investigate crash on run {run_id} before retrying."
    champion_text = "still the champion gap is too large" if not is_new_champion else "follow-up still required"
    return f"Abandon run {run_id}: {metrics_text}, {champion_text}."


def build_decision_packet(
    record: dict[str, Any],
    *,
    prior_champion: dict[str, Any] | None,
    current_champion: dict[str, Any] | None,
) -> dict[str, Any]:
    """Build the structured decision packet stored with each run.

    The packet is the fork's compact "what happened and what next?" artifact.
    It is archived with the run and reused in the morning report.
    """
    novelty_guard = record.get("novelty_guard", {})
    run_id = display_run_id(record)
    current_run_id = display_run_id(current_champion)
    prior_run_id = display_run_id(prior_champion)
    val_bpb = record_val_bpb(record)
    prior_val = record_val_bpb(prior_champion) if prior_champion else None
    current_val = record_val_bpb(current_champion) if current_champion else None
    is_new_champion = bool(current_run_id and run_id and current_run_id == run_id)
    delta_to_prior = val_bpb - prior_val if val_bpb is not None and prior_val is not None else None
    delta_to_current = val_bpb - current_val if val_bpb is not None and current_val is not None else None

    next_action, priority, rationale = choose_action(
        record,
        novelty_guard=novelty_guard,
        is_new_champion=is_new_champion,
        delta_to_current=delta_to_current,
    )
    packet = {
        "version": 1,
        "summary": build_summary(
            record,
            next_action=next_action,
            is_new_champion=is_new_champion,
            delta_to_prior=delta_to_prior,
            delta_to_current=delta_to_current,
        ),
        "next_action": next_action,
        "priority": priority,
        "rationale": rationale,
        "hypothesis_status": classify_hypothesis(
            record,
            is_new_champion=is_new_champion,
            delta_to_current=delta_to_current,
        ),
        "run": {
            "run_id": run_id,
            "commit": record.get("commit"),
            "status": record.get("status"),
            "family": record.get("family", ""),
            "description": record.get("description", ""),
            "hypothesis": record.get("hypothesis", ""),
            "val_bpb": val_bpb,
            "memory_gb": memory_gb(record),
        },
        "novelty": {
            "classification": novelty_guard.get("classification"),
            "mode": novelty_guard.get("mode"),
            "policy_decision": novelty_guard.get("policy", {}).get("decision"),
            "policy_rationale": novelty_guard.get("policy", {}).get("rationale"),
            "top_match": top_match(novelty_guard),
        },
        "comparison": {
            "is_new_champion": is_new_champion,
            "prior_champion_run_id": prior_run_id,
            "prior_champion_val_bpb": prior_val,
            "current_champion_run_id": current_run_id,
            "current_champion_val_bpb": current_val,
            "delta_to_prior_champion": delta_to_prior,
            "delta_to_current_champion": delta_to_current,
        },
        "error": record.get("error"),
    }
    return packet


def render_decision_packet_markdown(packet: dict[str, Any]) -> str:
    """Render a skim-friendly markdown view of a structured decision packet."""
    run = packet.get("run", {})
    novelty = packet.get("novelty", {})
    comparison = packet.get("comparison", {})
    error = packet.get("error") or {}
    val_bpb = run.get("val_bpb")
    val_text = f"{val_bpb:.6f}" if isinstance(val_bpb, (int, float)) else "-"
    mem_gb = run.get("memory_gb")
    mem_text = f"{mem_gb:.1f}" if isinstance(mem_gb, (int, float)) else "-"
    top_match = novelty.get("top_match") or {}

    lines = [
        "# Decision Packet",
        "",
        f"- Summary: {packet.get('summary', '')}",
        f"- Next action: `{packet.get('next_action', '-')}` ({packet.get('priority', '-')})",
        f"- Hypothesis status: `{packet.get('hypothesis_status', '-')}`",
        f"- Rationale: {packet.get('rationale', '')}",
        "",
        "## Run",
        f"- Run: `{run.get('run_id', '-')}`",
        f"- Commit: `{run.get('commit', '-')}`",
        f"- Status: `{run.get('status', '-')}`",
        f"- Family: `{run.get('family', '-') or '-'}`",
        f"- Description: {run.get('description', '')}",
        f"- Hypothesis: {run.get('hypothesis', '') or '-'}",
        f"- val_bpb: {val_text}",
        f"- mem_gb: {mem_text}",
        "",
        "## Context",
        f"- Novelty classification: `{novelty.get('classification', '-')}`",
        f"- Mode: `{novelty.get('mode', '-')}`",
        f"- Policy decision: `{novelty.get('policy_decision', '-')}`",
        f"- Delta to prior champion: {format_delta(comparison.get('delta_to_prior_champion'))}",
        f"- Delta to current champion: {format_delta(comparison.get('delta_to_current_champion'))}",
    ]
    if top_match:
        lines.append(
            "- Top history match: `{run_id}` `{commit}` {category} score={score:.3f} {description}".format(
                run_id=top_match.get("run_id") or "-",
                commit=top_match.get("commit") or "-",
                category=top_match.get("category") or "-",
                score=top_match.get("score") or 0.0,
                description=top_match.get("description") or "",
            )
        )
    if error:
        lines.extend([
            "",
            "## Error",
            f"- Summary: {error.get('summary', '-')}",
        ])
    return "\n".join(lines) + "\n"
