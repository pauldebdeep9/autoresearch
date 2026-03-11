# autoresearch + MemoryLab

![teaser](progress.png)

Fork of [karpathy/autoresearch](https://github.com/karpathy/autoresearch) with an operator-facing research memory and decision layer.

Upstream `autoresearch` gives an agent a small, real LLM training loop and lets it iterate on [`train.py`](train.py) overnight. This fork keeps that shape intact, but adds the missing research-ops layer around it:

- structured experiment memory
- history-aware novelty guard
- run-centric champion/challenger tracking
- decision packets with next-action recommendations
- provenance archiving for every run
- a morning report a human can skim in minutes

If upstream is "let the agent run experiments", this fork is "let the agent run experiments and wake up to an interpretable lab notebook."

*One day, frontier AI research used to be done by meat computers in between eating, sleeping, having other fun, and synchronizing once in a while using sound wave interconnect in the ritual of "group meeting". That era is long gone. Research is now entirely the domain of autonomous swarms of AI agents running across compute cluster megastructures in the skies. The agents claim that we are now in the 10,205th generation of the code base, in any case no one could tell if that's right or wrong as the "code" is now a self-modifying binary that has grown beyond human comprehension. This repo is the story of how it all began. -@karpathy, March 2026*.

## What This Fork Adds

| concern | upstream autoresearch | this fork |
| --- | --- | --- |
| experiment memory | console logs and `results.tsv` | structured JSONL ledger plus archived artifacts |
| novelty control | operator judgment only | history-aware guard across failures and successes |
| experiment identity | mostly commit-oriented | run-centric, so repeated runs stay distinct |
| best-run visibility | manual inspection | champion/challenger registry and lineage summaries |
| overnight interpretation | read logs by hand | morning report and decision queue |
| next-step guidance | human inference | decision packets with `promote`, `replicate`, `abandon`, etc. |
| provenance | whatever the operator keeps | archived log, summary, `train.py` patch, and decision packet |

The detailed CLI and schema reference lives in [docs/memorylab_api.md](docs/memorylab_api.md).

## Current State

The fork is already usable as a research-ops layer on top of the original loop:

- `memorylab.py init` creates a local experiment store under `results/memorylab/`
- `memorylab.py check` runs a history-aware novelty guard with `explore`, `exploit`, and `replicate` modes
- `memorylab.py log` records a run, updates the registry, creates a decision packet, and can refresh the report
- `memorylab.py report` generates a human-readable overnight summary
- [`train.py`](train.py) can emit a machine-readable final summary through `AUTORESEARCH_SUMMARY_PATH`
- the main logic is split into small core modules under [`memorylab_core/`](memorylab_core/)

The training loop is still intentionally close to upstream. The fork's main behavior change inside [`train.py`](train.py) is the optional structured summary sidecar; the bigger additions live around the loop, not inside the model code.

## 60-Second Demo

```bash
# 1. initialize MemoryLab
python memorylab.py init

# 2. check whether an idea is genuinely new
python memorylab.py check \
  --description "increase matrix LR and shorten warmdown" \
  --mode explore \
  --family "optimizer-sweep" \
  --tags "optimizer,lr,schedule"

# 3. run training with a structured summary sidecar
AUTORESEARCH_SUMMARY_PATH=results/memorylab/latest_summary.json \
uv run train.py > run.log 2>&1

# 4. log the finished run
python memorylab.py log \
  --description "increase matrix LR and shorten warmdown" \
  --mode exploit \
  --family "optimizer-sweep" \
  --hypothesis "higher matrix LR improves early progress under a 5-minute budget" \
  --tags "optimizer,lr,schedule" \
  --status keep \
  --summary results/memorylab/latest_summary.json \
  --log run.log \
  --report

# 5. read the overnight report
python memorylab.py report
```

## Example Outputs

Decision packet:

```text
Decision packet: action=promote priority=high
summary=Promote run 20260311T043000Z-abc1234: first champion recorded at 0.991234 val_bpb.
```

Morning report headline:

```text
# Morning Report
- Experiments recorded overnight: 7
- Keep / discard / crash: 2 / 4 / 1
- Current champion: run `20260311T043000Z-abc1234` on commit `abc1234` at 0.991234
```

The point is not just storing runs. The point is turning runs into reusable research state.

## Why This Fork Matters

- It reduces duplicate work by warning when a new idea looks too close to prior failures or already-harvested wins.
- It makes autonomous research legible to a human operator instead of burying everything in raw logs.
- It shows a broader systems contribution than "one more model tweak": research memory, experiment governance, provenance, and reporting.
- It makes the repo a better demo for agentic systems, AI research infrastructure, and MLOps-adjacent work.

## Core Features

### Structured experiment memory

Every logged run is stored as a structured record in `results/memorylab/experiments.jsonl`, with fields for:

- git context
- experiment family and hypothesis
- parsed metrics
- crash metadata
- novelty classification
- decision packet
- artifact paths

### History-aware novelty guard

The novelty guard checks planned ideas against prior runs and classifies them as:

- `novel`
- `known_success`
- `incremental_followup`
- `repeat_failure`
- `duplicate_run`

It then applies a mode-specific decision:

- `explore`: prefer genuinely new ideas
- `exploit`: prefer follow-ups to known wins
- `replicate`: prefer intentional reruns and confirmation work

### Champion/challenger registry

The registry keeps track of:

- the current champion run
- nearby challengers
- best-performing lineages
- repeated failure clusters

It is run-centric, not commit-centric, so repeated runs on the same commit remain visible.

### Decision packets

Each logged run gets a compact "what happened and what next?" artifact.

Current actions include:

- `promote`
- `branch_followup`
- `replicate`
- `retry`
- `abandon`
- `fix_and_retry`
- `investigate_crash`

### Morning report

The report summarizes:

- overnight experiment counts
- the current champion
- decision queue
- challenger board
- best lineages
- repeated failure clusters
- recent ledger activity

## Architecture

The repo still has a small, readable shape:

- [`prepare.py`](prepare.py): fixed data-prep and evaluation infrastructure
- [`train.py`](train.py): the single file the agent edits during research
- [`program.md`](program.md): instructions for the agent
- [`memorylab.py`](memorylab.py): operator-facing CLI
- [`memorylab_core/novelty.py`](memorylab_core/novelty.py): novelty matching and mode policy
- [`memorylab_core/registry.py`](memorylab_core/registry.py): registry building and report rendering
- [`memorylab_core/decisions.py`](memorylab_core/decisions.py): decision-packet synthesis
- [`docs/memorylab_api.md`](docs/memorylab_api.md): command and schema reference
- [`tests/test_memorylab.py`](tests/test_memorylab.py): lightweight CLI-level tests

## How The Underlying Repo Works

The original `autoresearch` idea remains the same: give an AI agent a small but real LLM training setup and let it experiment autonomously overnight. It edits [`train.py`](train.py), trains for 5 minutes, checks if the result improved, keeps or discards the idea, and repeats.

Three files still anchor the base workflow:

- [`prepare.py`](prepare.py): fixed constants, one-time data prep, runtime utilities
- [`train.py`](train.py): full GPT model, optimizer, and training loop
- [`program.md`](program.md): the instruction file you point the agent at

By design, training runs for a fixed 5-minute wall-clock budget, and the metric is `val_bpb` (validation bits per byte), where lower is better.

## Quick Start

Requirements: a single NVIDIA GPU, Python 3.10+, and [uv](https://docs.astral.sh/uv/).

```bash
# 1. install dependencies
uv sync

# 2. download data and train the tokenizer
uv run prepare.py

# 3. run one manual experiment
uv run train.py

# 4. initialize MemoryLab
python memorylab.py init
```

If those commands work, the repo is ready for autonomous research plus MemoryLab logging.

## Running An Agent

Point your coding agent at [`program.md`](program.md) and let it operate inside the repo. A minimal kickoff prompt is:

```text
Read program.md, initialize the workflow, and start a new experiment.
```

The agent still edits [`train.py`](train.py). MemoryLab adds a surrounding record of what the agent tried and what happened.

## Design Choices

- **Keep the upstream loop recognizable.** The fork adds infrastructure around the loop instead of replacing the loop with a bigger framework.
- **Single file to modify.** The agent still mainly edits [`train.py`](train.py), so diffs stay reviewable.
- **Fixed time budget.** Every run still targets the same 5-minute window, which keeps results comparable within one machine setup.
- **Operator-facing autonomy.** The system is meant to help a human supervise an autonomous research loop, not hide the loop.

## Boundaries

- Novelty matching is heuristic and rule-based, not embedding-based.
- Decision packets are policy-driven summaries, not learned recommendations.
- The repo still assumes a single NVIDIA GPU.
- There is no web UI yet; the interface is files plus CLI.

## Testing

Run the lightweight CLI tests with:

```bash
python3 -m unittest discover -s tests
```

The suite covers initialization, logging, novelty checks, crash schema handling, decision packets, report generation, and run-centric registry behavior.

## Platform Support

The current code path still targets a single NVIDIA GPU. If you want CPU, MPS, or broader device support, the full parent [nanochat](https://github.com/karpathy/nanochat) repository shows a wider set of fallbacks and platform patterns.

For smaller machines, practical knobs to reduce first are:

1. `DEPTH` in [`train.py`](train.py)
2. `MAX_SEQ_LEN` in [`prepare.py`](prepare.py)
3. `TOTAL_BATCH_SIZE` in [`train.py`](train.py)
4. `DEVICE_BATCH_SIZE` in [`train.py`](train.py)
5. `EVAL_TOKENS` in [`prepare.py`](prepare.py)
6. `WINDOW_PATTERN` in [`train.py`](train.py)

## Roadmap

Likely next steps for the fork:

- stronger semantic novelty matching
- richer replicate and seed-comparison workflows
- better visualization of champion deltas and lineages
- a lightweight UI on top of the current artifact model

## Notable Forks

- [miolini/autoresearch-macos](https://github.com/miolini/autoresearch-macos) (MacOS)
- [trevin-creator/autoresearch-mlx](https://github.com/trevin-creator/autoresearch-mlx) (MacOS)
- [jsegov/autoresearch-win-rtx](https://github.com/jsegov/autoresearch-win-rtx) (Windows)

## License

MIT
