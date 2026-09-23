"""Context compaction helpers extracted so they can be unit-tested without the engine."""

from __future__ import annotations


def estimate_tokens(text: str) -> int:
    """Cheap mixed CJK/ASCII estimator. CJK ≈ 1.5 chars/token, ASCII ≈ 4."""
    if not text:
        return 0
    cjk = sum(1 for ch in text if "\u4e00" <= ch <= "\u9fff")
    other = len(text) - cjk
    return max(1, int(cjk / 1.5 + other / 4))


def apply_compact_delta(current: str, snapshot: str, compacted: str) -> str:
    """Apply a background compression result without dropping steps added after the snapshot.

    The compact thread works on `snapshot`. The main loop may have appended new
    Thought/Action/Observation blocks since then. If `current` still starts with
    `snapshot`, keep the suffix (the new steps) after the compacted text.
    """
    if snapshot and current.startswith(snapshot):
        return compacted + current[len(snapshot) :]
    return compacted
