"""Fonte local de documentos.

Substitui o storage remoto do projeto original: em vez de baixar os
editais de um bucket, o Decifra lê os arquivos de uma pasta local
(``data/editais`` por padrão).

TODO: quando sair da PoC, trocar essa leitura de volume local por
Object Storage (MinIO/S3) -- ``iter_documents`` é o ponto de troca;
o resto do pipeline (`ingestion/pipeline.py`) já consome só
``(nome, bytes)`` e não muda.
"""

from __future__ import annotations

from collections.abc import Iterator
from pathlib import Path

_ALLOWED = {".pdf", ".docx", ".doc"}


def iter_documents(data_dir: str) -> Iterator[tuple[str, bytes]]:
    """Percorre a pasta e devolve cada documento como bytes.

    Parameters
    ----------
    data_dir : str
        Caminho da pasta com os editais.

    Yields
    ------
    tuple[str, bytes]
        Nome do arquivo e o conteúdo binário.
    """
    root = Path(data_dir)
    if not root.exists():
        raise FileNotFoundError(f"Pasta de editais não encontrada: {root}")
    for path in sorted(root.rglob("*")):
        if path.is_file() and path.suffix.lower() in _ALLOWED:
            yield path.name, path.read_bytes()
