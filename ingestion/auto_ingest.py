"""
Ingestão de um único edital descoberto por uma fonte automática.

Cada edital é tratado como uma unidade independente: uma falha (download
quebrado, PDF corrompido) nunca deve impedir os outros de serem
processados. Usa retry com backoff exponencial (mesmo padrão de
``retrieval/generate.py``); se esgotar as tentativas, registra como
dead-letter no log e devolve status "failed" em vez de levantar.
"""

from __future__ import annotations

import logging
import time
from collections.abc import Callable

from ingestion.fetch import baixar_conteudo
from ingestion.pipeline import ingest_document
from ingestion.sources.base import EditalRef
from ingestion.weaviate_store import WeaviateStore

logger = logging.getLogger(__name__)

_DELAYS_SEGUNDOS = (2, 4, 8)


def _nome_arquivo(ref: EditalRef) -> str:
    ultimo_segmento = ref.pdf_url.rsplit("/", 1)[-1].split("?")[0]
    if ultimo_segmento.lower().endswith((".pdf", ".docx", ".doc")):
        return ultimo_segmento
    return f"{ref.titulo}.pdf"


def ingerir_edital(
    ref: EditalRef,
    store: WeaviateStore,
    *,
    sleep_fn: Callable[[float], None] = time.sleep,
) -> dict:
    """Baixa e ingere um edital, isolando falhas.

    Tenta até `len(_DELAYS_SEGUNDOS) + 1` vezes com backoff exponencial.
    Nunca levanta exceção — na última falha, loga como dead-letter e
    devolve ``{"status": "failed", ...}``.
    """
    file_name = _nome_arquivo(ref)
    tentativas = len(_DELAYS_SEGUNDOS) + 1

    ultimo_erro: Exception | None = None
    for tentativa in range(1, tentativas + 1):
        try:
            conteudo = baixar_conteudo(ref.pdf_url)
            resultado = ingest_document(
                file_name,
                conteudo,
                store,
                bucket="ifsc_site" if ref.pdf_url.startswith("http") else "local",
                storage_path=ref.pdf_url,
                status=ref.status,
            )
            resultado.setdefault("file_name", file_name)
            return resultado
        except Exception as exc:  # noqa: BLE001 - isolamento por edital é o objetivo
            ultimo_erro = exc
            if tentativa <= len(_DELAYS_SEGUNDOS):
                logger.warning(
                    "Falha ao ingerir '%s' (tentativa %d/%d): %s. Nova tentativa em %ds.",
                    ref.titulo,
                    tentativa,
                    tentativas,
                    exc,
                    _DELAYS_SEGUNDOS[tentativa - 1],
                )
                sleep_fn(_DELAYS_SEGUNDOS[tentativa - 1])

    logger.error(
        "DEAD-LETTER: '%s' (%s) falhou após %d tentativas: %s",
        ref.titulo,
        ref.pdf_url,
        tentativas,
        ultimo_erro,
    )
    return {
        "status": "failed",
        "file_name": file_name,
        "pdf_url": ref.pdf_url,
        "erro": str(ultimo_erro),
    }
