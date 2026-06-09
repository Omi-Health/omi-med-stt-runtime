from __future__ import annotations

import re


def normalize_ws(text: str) -> str:
    return re.sub(r"\s+", " ", text).strip()


def merge_overlap(existing: str, addition: str, max_overlap_words: int = 80) -> str:
    existing = normalize_ws(existing)
    addition = normalize_ws(addition)
    if not existing:
        return addition
    if not addition:
        return existing
    a = existing.split()
    b = addition.split()
    max_n = min(max_overlap_words, len(a), len(b))
    for n in range(max_n, 0, -1):
        if [w.lower() for w in a[-n:]] == [w.lower() for w in b[:n]]:
            return normalize_ws(" ".join(a + b[n:]))
    return normalize_ws(existing + " " + addition)


def merge_transcripts(parts: list[str]) -> str:
    merged = ""
    for part in parts:
        merged = merge_overlap(merged, part)
    return merged
