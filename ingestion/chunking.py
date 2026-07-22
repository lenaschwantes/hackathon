"""Chunking por tokens (tiktoken).

Alternativa leve ao chunker hierárquico do docling para a PoC: divide o
texto limpo em janelas de tokens com sobreposição. Suficiente para o
demo; o chunker do docling pode entrar depois sem mudar o resto.
"""

from __future__ import annotations

import tiktoken

from config.settings import settings

_enc = tiktoken.get_encoding("cl100k_base")


def chunk_text(
    text: str, size: int | None = None, overlap: int | None = None
) -> list[str]:
    """Divide o texto em janelas de tokens com sobreposição.

    Parameters
    ----------
    text : str
        Texto limpo do edital.
    size : int, optional
        Tokens por chunk. Default: ``settings.chunk_size``.
    overlap : int, optional
        Tokens de sobreposição entre chunks. Default: ``settings.chunk_overlap``.

    Returns
    -------
    list[str]
        Lista de trechos de texto.
    """
    size = size or settings.chunk_size
    overlap = overlap if overlap is not None else settings.chunk_overlap
    tokens = _enc.encode(text or "")
    if not tokens:
        return []

    step = max(1, size - overlap)
    chunks: list[str] = []
    for start in range(0, len(tokens), step):
        window = tokens[start : start + size]
        chunks.append(_enc.decode(window))
        if start + size >= len(tokens):
            break
    return chunks
