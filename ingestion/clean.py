from __future__ import annotations

import re
import unicodedata


def clean_text(text: str) -> str:
    """Normalize extracted text before persistence (ETL only, no chunking)."""
    text = unicodedata.normalize("NFC", text or "")
    text = text.replace("\r\n", "\n").replace("\r", "\n")

    text = "".join(
        c
        for c in text
        if c in ("\n", "\t") or not unicodedata.category(c).startswith("C")
    )

    lines: list[str] = []
    for line in text.split("\n"):
        line = re.sub(r"[ \t]+", " ", line).strip()
        if not line:
            if lines and lines[-1] != "":
                lines.append("")
            continue
        lines.append(line)

    cleaned = "\n".join(lines)
    cleaned = re.sub(r"\n{3,}", "\n\n", cleaned)
    return cleaned.strip()
