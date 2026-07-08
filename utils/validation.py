from __future__ import annotations

from pathlib import Path

ALLOWED_SUFFIXES = frozenset({".pdf", ".docx", ".doc", ".pptx"})


def assert_allowed_filename(filename: str) -> str:
    suffix = Path(filename).suffix.lower()
    if suffix not in ALLOWED_SUFFIXES:
        allowed = ", ".join(sorted(ALLOWED_SUFFIXES))
        raise ValueError(f"Unsupported file type '{suffix}'. Allowed: {allowed}")
    return suffix


def assert_has_extractable_text(text: str, *, min_chars: int) -> int:
    cleaned = (text or "").strip()
    length = len(cleaned)
    if length < min_chars:
        raise ValueError(
            f"Extracted text too short ({length} chars). "
            f"Minimum required: {min_chars}. Check document quality or format."
        )
    return length
