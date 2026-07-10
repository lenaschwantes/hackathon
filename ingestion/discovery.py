"""
Detecção incremental: filtra os editais que a fonte listou mas que ainda
não foram processados, pra não reprocessar nem duplicar em runs seguintes.
"""

from __future__ import annotations

from collections.abc import Callable

from ingestion.sources.base import EditalRef, EditalSource


def descobrir_novos(
    source: EditalSource, ja_conhecido: Callable[[str], bool]
) -> list[EditalRef]:
    """Lista os editais da fonte e devolve só os que `ja_conhecido` não reconhece.

    Parameters
    ----------
    source : EditalSource
        Fonte a consultar.
    ja_conhecido : Callable[[str], bool]
        Recebe o `pdf_url` de um edital e diz se ele já foi processado
        antes (ex.: já existe no Weaviate por `storage_path`).
    """
    refs = source.list_editais()
    return [ref for ref in refs if not ja_conhecido(ref.pdf_url)]
