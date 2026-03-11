"""History-aware novelty detection for MemoryLab.

This module is one of the main additions in the fork. Upstream autoresearch
has a tight experiment loop but no memory of whether an idea is genuinely new,
close to a prior success, or a likely repeat of a failed branch. MemoryLab uses
this module to classify new ideas before a run is launched and to store the
same classification alongside completed runs.

The matcher is intentionally heuristic and operator-friendly:
- normalize free-form experiment text into a canonical token space
- collapse common aliases such as "learning rate" and "step size"
- blend lexical similarity with concept overlap and number overlap
- classify against all prior runs, not only failures
- apply a policy mode (`explore`, `exploit`, `replicate`) on top

The goal is not perfect semantic search. The goal is to prevent obvious
duplicate work and make the research loop more legible.
"""

from __future__ import annotations

import re
from difflib import SequenceMatcher
from typing import Any


# Normalization is deliberately opinionated toward this repo's experiment space.
# These tables turn varied operator phrasing into a narrower internal vocabulary
# so "raise matrix learning rate" and "increase matrix lr" land near each other.
STOPWORDS = {
    "a", "an", "and", "are", "as", "at", "baseline", "be", "by", "for", "from",
    "if", "in", "into", "is", "it", "keep", "of", "on", "or", "the", "to", "try",
    "up", "use", "with",
}
KEEP_SHORT_TOKENS = {"lr", "kv", "ve", "fa", "mlp", "oom"}
TOKEN_RE = re.compile(r"[a-z0-9_]+")
NUMBER_RE = re.compile(r"\b\d+(?:\.\d+)?\b")
PHRASE_ALIASES = {
    "learning rate": "lr",
    "step size": "lr",
    "matrix learning rate": "matrix_lr",
    "matrix lr": "matrix_lr",
    "embedding learning rate": "embedding_lr",
    "unembedding learning rate": "unembedding_lr",
    "scalar learning rate": "scalar_lr",
    "value embedding": "value_embedding",
    "value embeddings": "value_embedding",
    "out of memory": "oom",
    "cuda out of memory": "oom",
    "warm down": "warmdown",
    "cool down": "warmdown",
    "gradient accumulation": "grad_accum",
    "batch size": "batch_size",
    "window pattern": "window_pattern",
    "attention window": "window_pattern",
    "head dimension": "head_dim",
    "model width": "width",
}
TOKEN_ALIASES = {
    "increase": "up",
    "increasing": "up",
    "raise": "up",
    "higher": "up",
    "boost": "up",
    "larger": "up",
    "bigger": "up",
    "decrease": "down",
    "decreasing": "down",
    "lower": "down",
    "reduce": "down",
    "reducing": "down",
    "shorten": "down",
    "smaller": "down",
    "cooldown": "warmdown",
    "schedule": "sched",
    "optimizer": "optim",
    "activation": "act",
    "geglu": "gated_act",
    "swiglu": "gated_act",
    "glu": "gated_act",
}
CONCEPT_PATTERNS = [
    ("lr", re.compile(r"\b(matrix_lr|embedding_lr|unembedding_lr|scalar_lr|lr)\b")),
    ("schedule", re.compile(r"\b(warmup|warmdown|sched)\b")),
    ("memory", re.compile(r"\b(oom|memory|vram)\b")),
    ("architecture", re.compile(r"\b(width|depth|head_dim|heads|attention|window_pattern|act|mlp|value_embedding)\b")),
    ("batching", re.compile(r"\b(batch_size|grad_accum|tokens|sequence|context)\b")),
    ("optimizer", re.compile(r"\b(muon|adamw|optim|weight_decay|beta)\b")),
]
NOVELTY_PRIORITY = {
    "duplicate_run": 0,
    "repeat_failure": 1,
    "incremental_followup": 2,
    "known_success": 3,
    "novel": 4,
}
NOVELTY_ORDER = ("duplicate_run", "repeat_failure", "incremental_followup", "known_success")
MODE_DEFAULT = "exploit"
MODE_ADJUSTMENTS = {
    "explore": -0.05,
    "exploit": 0.0,
    "replicate": -0.08,
}
MODE_POLICY = {
    "explore": {
        "novel": ("allow", "novel enough for exploration"),
        "duplicate_run": ("block", "duplicate work is not exploration"),
        "repeat_failure": ("block", "repeating a failed idea is not exploratory enough"),
        "incremental_followup": ("block", "follow-up work belongs in exploit mode"),
        "known_success": ("block", "reusing a known success belongs in exploit mode"),
    },
    "exploit": {
        "novel": ("caution", "novel work may be fine, but it is not targeted exploitation"),
        "duplicate_run": ("block", "exact duplicates belong in replicate mode"),
        "repeat_failure": ("block", "prior failures are poor exploitation candidates"),
        "incremental_followup": ("allow", "close to a known success in the same family"),
        "known_success": ("allow", "close to a prior successful idea"),
    },
    "replicate": {
        "novel": ("block", "nothing close enough in history to replicate"),
        "duplicate_run": ("allow", "matches an existing run closely enough to replicate"),
        "repeat_failure": ("block", "repeating a failed idea is not a useful replication target"),
        "incremental_followup": ("allow", "close enough to a prior run in the same family to replicate"),
        "known_success": ("allow", "close enough to a prior successful run to replicate"),
    },
}


def normalize_inline_text(text: str) -> str:
    """Collapse internal whitespace without changing word order."""
    return " ".join(text.split())


def normalize_tags(raw_tags: str | None) -> list[str]:
    """Parse user-facing comma-separated tags into stable deduplicated tokens."""
    if not raw_tags:
        return []
    parts = []
    for item in raw_tags.split(","):
        tag = re.sub(r"[^a-z0-9_-]+", "-", item.strip().lower()).strip("-")
        if tag and tag not in parts:
            parts.append(tag)
    return parts


def apply_aliases(text: str) -> str:
    """Replace known multi-token phrases before tokenization.

    Phrase aliasing happens before tokenization because many of the useful
    experiment concepts in this repo are naturally multi-word phrases.
    """
    lowered = text.lower().replace("-", " ")
    for source, target in sorted(PHRASE_ALIASES.items(), key=lambda item: len(item[0]), reverse=True):
        lowered = lowered.replace(source, target)
    return lowered


def canonicalize_token(token: str) -> str:
    """Map single tokens into the internal canonical vocabulary."""
    return TOKEN_ALIASES.get(token, token)


def normalize_text(text: str) -> str:
    """Normalize arbitrary free text into a canonical comparison string."""
    aliased = apply_aliases(text)
    tokens = [canonicalize_token(token) for token in TOKEN_RE.findall(aliased)]
    return " ".join(tokens)


def tokenize(text: str) -> list[str]:
    """Tokenize normalized text while dropping low-signal stopwords."""
    tokens = []
    for token in TOKEN_RE.findall(normalize_text(text)):
        if token in STOPWORDS:
            continue
        if len(token) > 2 or token in KEEP_SHORT_TOKENS:
            tokens.append(token)
    return tokens


def extract_concepts(text: str) -> set[str]:
    """Extract coarse experiment concepts and explicit numeric markers."""
    normalized = normalize_text(text)
    concepts = {name for name, pattern in CONCEPT_PATTERNS if pattern.search(normalized)}
    for number in NUMBER_RE.findall(normalized):
        concepts.add(f"num:{number}")
    return concepts


def experiment_text(record: dict[str, Any]) -> str:
    """Build the comparison text for a stored experiment record."""
    tags = " ".join(record.get("tags", []))
    parts = [
        record.get("description", ""),
        record.get("hypothesis", ""),
        record.get("family", ""),
        tags,
    ]
    return " ".join(part for part in parts if part).strip()


def build_probe_text(description: str, tags: list[str], family: str = "", hypothesis: str = "") -> str:
    """Build the comparison text for a proposed future experiment."""
    parts = [description, hypothesis, family, " ".join(tags)]
    return " ".join(part for part in parts if part).strip()


def normalize_mode(mode: str | None) -> str:
    """Normalize the novelty policy mode, defaulting to exploit."""
    if not mode:
        return MODE_DEFAULT
    return mode.lower()


def effective_threshold(threshold: float, mode: str) -> float:
    """Apply mode-specific threshold tuning on top of the base similarity threshold."""
    adjusted = threshold + MODE_ADJUSTMENTS.get(normalize_mode(mode), 0.0)
    return max(0.0, min(0.99, adjusted))


def similarity_score(a: str, b: str) -> float:
    """Blend lexical and concept similarity into one operator-friendly score.

    The score is intentionally interpretable rather than learned:
    - sequence ratio catches near-verbatim rewrites
    - token overlap catches reordered phrasing
    - concept overlap catches idea-level similarity
    - number overlap catches hyperparameter repeats
    """
    norm_a = normalize_text(a)
    norm_b = normalize_text(b)
    if not norm_a or not norm_b:
        return 0.0

    seq_ratio = SequenceMatcher(None, norm_a, norm_b).ratio()
    tokens_a = set(tokenize(norm_a))
    tokens_b = set(tokenize(norm_b))
    token_jaccard = len(tokens_a & tokens_b) / len(tokens_a | tokens_b) if (tokens_a or tokens_b) else 0.0

    concepts_a = extract_concepts(norm_a)
    concepts_b = extract_concepts(norm_b)
    concept_overlap = len(concepts_a & concepts_b) / len(concepts_a | concepts_b) if (concepts_a or concepts_b) else 0.0

    numbers_a = set(NUMBER_RE.findall(norm_a))
    numbers_b = set(NUMBER_RE.findall(norm_b))
    number_overlap = len(numbers_a & numbers_b) / len(numbers_a | numbers_b) if (numbers_a or numbers_b) else 0.0

    return round(0.3 * seq_ratio + 0.35 * token_jaccard + 0.25 * concept_overlap + 0.1 * number_overlap, 3)


def is_duplicate_probe(probe_text: str, record_text: str, score: float) -> bool:
    """Detect effectively identical ideas, even if formatting differs slightly."""
    normalized_probe = normalize_text(probe_text)
    normalized_record = normalize_text(record_text)
    if normalized_probe and normalized_probe == normalized_record:
        return True
    probe_tokens = set(tokenize(probe_text))
    record_tokens = set(tokenize(record_text))
    return score >= 0.96 and probe_tokens == record_tokens


def classify_match(record: dict[str, Any], family: str, probe_text: str, score: float) -> str:
    """Classify a single historical match into the novelty taxonomy."""
    if is_duplicate_probe(probe_text, experiment_text(record), score):
        return "duplicate_run"
    status = record.get("status")
    if status in {"discard", "crash"}:
        return "repeat_failure"
    if status == "keep" and family and record.get("family") == family:
        return "incremental_followup"
    if status == "keep":
        return "known_success"
    return "novel"


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
    """Classify a proposed idea against all prior runs.

    Returns a compact summary that is stored in the run ledger and also printed
    by `memorylab.py check`. The result is intentionally explicit so operators
    can see both the top classification and the nearby historical evidence.
    """
    probe_text = build_probe_text(description, tags, family=family, hypothesis=hypothesis)
    matches = []
    for record in records:
        record_text = experiment_text(record)
        score = similarity_score(probe_text, record_text)
        category = classify_match(record, family, probe_text, score)
        same_family = bool(family and record.get("family") == family)
        # Same-family follow-up work gets a slightly wider catchment area so the
        # guard can recognize "this is more of the same line" without requiring
        # an almost identical description.
        if score < threshold and not is_duplicate_probe(probe_text, record_text, score):
            if not (same_family and score >= max(0.0, threshold - 0.08)):
                continue
        metrics = record.get("metrics", {})
        matches.append({
            "category": category,
            "score": score,
            "commit": record.get("commit"),
            "run_id": record.get("run_id"),
            "family": record.get("family", ""),
            "status": record.get("status"),
            "val_bpb": metrics.get("val_bpb"),
            "description": record.get("description", ""),
            "tags": record.get("tags", []),
        })

    matches.sort(
        key=lambda row: (
            NOVELTY_PRIORITY.get(row["category"], 99),
            -row["score"],
            row["val_bpb"] if row["val_bpb"] is not None else 999,
        )
    )

    counts = {category: 0 for category in NOVELTY_ORDER}
    for match in matches:
        if match["category"] in counts:
            counts[match["category"]] += 1

    classification = "novel"
    for category in NOVELTY_ORDER:
        if counts[category]:
            classification = category
            break

    return {
        "classification": classification,
        "probe_text": probe_text,
        "threshold": threshold,
        "match_count": len(matches),
        "counts": counts,
        "top_matches": matches[:limit],
    }


def evaluate_policy(classification: str, mode: str) -> dict[str, str]:
    """Map a novelty classification into a mode-specific allow/caution/block decision."""
    normalized_mode = normalize_mode(mode)
    decision, rationale = MODE_POLICY[normalized_mode][classification]
    return {
        "mode": normalized_mode,
        "decision": decision,
        "rationale": rationale,
    }


def find_similar_failures(
    records: list[dict[str, Any]],
    description: str,
    tags: list[str],
    *,
    threshold: float,
    limit: int = 3,
) -> list[dict[str, Any]]:
    """Backward-compatible helper for callers that only care about failures."""
    history = classify_history(
        records,
        description,
        tags,
        threshold=threshold,
        limit=max(limit, 5),
    )
    matches = [
        match for match in history["top_matches"]
        if match["category"] in {"duplicate_run", "repeat_failure"}
    ]
    return matches[:limit]
