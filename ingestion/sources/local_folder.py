"""
Fonte local de editais — o fallback manual atual.

Lista os PDFs/DOCX de uma pasta local (``data/editais`` por padrão), sem
depender do site do IFSC estar no ar. Sempre reportado como "aberto": um
edital colocado manualmente na pasta é presumidamente vigente.
"""

from __future__ import annotations

from pathlib import Path

from ingestion.sources.base import EditalRef, EditalSource

_ALLOWED = {".pdf", ".docx", ".doc"}


class LocalFolderSource(EditalSource):
    def __init__(self, data_dir: str):
        self._data_dir = data_dir

    def list_editais(self) -> list[EditalRef]:
        root = Path(self._data_dir)
        if not root.exists():
            return []

        refs = []
        for path in sorted(root.rglob("*")):
            if path.is_file() and path.suffix.lower() in _ALLOWED:
                refs.append(
                    EditalRef(titulo=path.stem, pdf_url=str(path), status="aberto")
                )
        return refs
