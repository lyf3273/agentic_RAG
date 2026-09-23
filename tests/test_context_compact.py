from context_compact import apply_compact_delta, estimate_tokens


def test_apply_compact_keeps_suffix_appended_after_snapshot():
    snapshot = "Thought: a\nObservation: old\n\n"
    current = snapshot + "Thought: b\nObservation: NEW\n\n"
    compacted = "[推理摘要]\n已查完 a\n\n"
    merged = apply_compact_delta(current, snapshot, compacted)
    assert "NEW" in merged
    assert merged.startswith(compacted)


def test_apply_compact_without_prefix_falls_back():
    merged = apply_compact_delta("other", "snap", "compacted")
    assert merged == "compacted"


def test_estimate_tokens_cjk_heavier_than_ascii():
    zh = estimate_tokens("应急响应" * 10)
    en = estimate_tokens("abcd" * 20)
    assert zh > 10
    assert en >= 1
