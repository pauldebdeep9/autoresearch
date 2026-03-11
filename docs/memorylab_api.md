# MemoryLab API Reference

`MemoryLab` is the main operator-facing addition in this fork. Upstream `autoresearch` gives an agent a tight experiment loop around [`train.py`](../train.py) and [`program.md`](../program.md). This fork adds a research-operations layer on top of that loop so experiments become legible, searchable, and resumable.

## What This Fork Adds Over Upstream

Upstream behavior:
- run short experiments against `train.py`
- inspect `val_bpb`
- keep or discard the resulting commit

This fork adds:
- a structured experiment ledger
- history-aware novelty checks against prior failures and successes
- policy modes for exploration, exploitation, and replication
- a run-centric champion/challenger registry
- decision packets with recommended next actions
- archived provenance for each run
- a human-readable morning report

The goal is not to replace the upstream research loop. The goal is to make the loop observable and reusable by a human operator or a future agent.

## Command Surface

Primary CLI entrypoint: [`memorylab.py`](../memorylab.py)

### `python memorylab.py init`

Initializes the local MemoryLab store:
- `results/memorylab/experiments.jsonl`
- `results/memorylab/champion_challenger.json`
- `results/memorylab/reports/`
- `results/memorylab/runs/`
- compatibility `results.tsv`

### `python memorylab.py check`

History-aware novelty guard for a planned idea.

Inputs:
- `--description`
- `--tags`
- `--family`
- `--mode explore|exploit|replicate`
- `--threshold`
- `--limit`
- `--fail-on-similar`

Outputs:
- a novelty classification
- a policy decision
- the top matching prior runs

Novelty classifications:
- `novel`
- `known_success`
- `incremental_followup`
- `repeat_failure`
- `duplicate_run`

Policy decisions:
- `allow`
- `caution`
- `block`

### `python memorylab.py log`

Records a completed run, updates MemoryLab state, optionally refreshes the morning report.

Inputs:
- run description and tags
- `status` in `keep|discard|crash`
- optional `family`, `hypothesis`, and `notes`
- `--summary` path from `AUTORESEARCH_SUMMARY_PATH`
- `--log` path for parsing/archiving
- novelty `--mode`

Side effects:
- appends a JSONL ledger row
- appends a compatibility row to `results.tsv`
- rebuilds the run-centric registry
- synthesizes a decision packet
- archives run artifacts under `results/memorylab/runs/<run_id>/`

### `python memorylab.py report`

Renders a human-readable morning report from the current ledger.

Outputs:
- `results/memorylab/reports/latest.md`
- `results/memorylab/reports/<timestamp>.md`

## Data Model

### Experiment ledger entry

File: `results/memorylab/experiments.jsonl`

High-level fields:
- `run_id`
- `timestamp_utc`
- `branch`
- `commit`
- `parent_commit`
- `family`
- `status`
- `description`
- `hypothesis`
- `tags`
- `notes`
- `metrics`
- `error`
- `artifacts`
- `novelty_guard`
- `decision_packet`

Important semantics:
- `status=crash` stores nullable structured metrics when no measurement exists
- `results.tsv` remains compatibility-oriented even when the JSONL schema is richer
- records are run-centric, so multiple runs on the same commit remain distinct

### `metrics`

Produced from the structured training sidecar or parsed from `run.log`.

Typical fields:
- `val_bpb`
- `training_seconds`
- `total_seconds`
- `peak_vram_mb`
- `mfu_percent`
- `total_tokens_M`
- `num_steps`
- `num_params_M`
- `depth`

### `error`

Only populated for crash runs.

Fields:
- `summary`
- `tail`
- `source`

### `novelty_guard`

Carries both the raw history classification and the mode-specific decision layer.

Fields:
- `classification`
- `probe_text`
- `threshold`
- `match_count`
- `counts`
- `top_matches`
- `mode`
- `effective_threshold`
- `policy`

### `decision_packet`

Decision packets are the fork’s main synthesis layer.

Purpose:
- capture what happened
- describe what it means
- recommend what should happen next

Fields:
- `summary`
- `next_action`
- `priority`
- `rationale`
- `hypothesis_status`
- `run`
- `novelty`
- `comparison`
- `error`

Current `next_action` values:
- `promote`
- `branch_followup`
- `replicate`
- `retry`
- `abandon`
- `fix_and_retry`
- `investigate_crash`

## Core Modules

### [`memorylab_core/novelty.py`](../memorylab_core/novelty.py)

Responsibilities:
- normalize free-text experiment ideas
- apply aliasing and concept extraction
- score similarity between a new idea and prior runs
- classify a proposal against history
- apply mode-specific novelty policy

### [`memorylab_core/registry.py`](../memorylab_core/registry.py)

Responsibilities:
- choose the current best run
- derive lineages from commit ancestry
- build challenger and lineage summaries
- cluster repeated failures
- render the morning report

### [`memorylab_core/decisions.py`](../memorylab_core/decisions.py)

Responsibilities:
- compare the current run to prior and current champions
- interpret novelty and crash state
- emit a concise next-action recommendation
- render a skim-friendly markdown packet

## Training Summary Sidecar

File producer: [`train.py`](../train.py)

Environment variable:
- `AUTORESEARCH_SUMMARY_PATH`

If set, `train.py` writes a machine-readable JSON summary of the final run metrics. This is the bridge between the upstream training loop and the new MemoryLab workflow. It lets the fork log experiments without scraping only human-oriented console output.

## Recommended Operator Flow

1. Run `python memorylab.py check` before editing `train.py`.
2. Run training with `AUTORESEARCH_SUMMARY_PATH=...`.
3. Log the run with `python memorylab.py log`.
4. Read the generated decision packet.
5. Use the morning report to review overnight progress.

This is the core value proposition of the fork: the repo does not just run experiments anymore, it now keeps enough structured memory to guide the next experiment.
