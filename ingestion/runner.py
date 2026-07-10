"""
Orquestra um ciclo de ingestão automática: descobre editais novos na fonte
e ingere cada um, isoladamente e em sequência (sem paralelismo agressivo,
respeitando o site).

Substitui o "Celery beat" por um agendador simples: `main()` roda um ciclo
único (`--once`) ou faz loop com `time.sleep` entre ciclos (`--loop`).
"""

from __future__ import annotations

import argparse
import logging
import time

from config.settings import settings
from ingestion.auto_ingest import ingerir_edital
from ingestion.discovery import descobrir_novos
from ingestion.pipeline import raw_collection_name
from ingestion.sources.base import EditalSource
from ingestion.sources.fallback import FallbackEditalSource
from ingestion.sources.ifsc_crawler import IFSCCrawler
from ingestion.sources.local_folder import LocalFolderSource
from ingestion.weaviate_store import WeaviateStore

logger = logging.getLogger(__name__)

_PAUSA_ENTRE_DOWNLOADS_SEGUNDOS = 1.0


def fonte_padrao() -> EditalSource:
    """Site do IFSC como fonte primária, pasta local como fallback."""
    return FallbackEditalSource(
        primaria=IFSCCrawler(),
        fallback=LocalFolderSource(settings.data_dir),
    )


def executar_ciclo(source: EditalSource, store: WeaviateStore) -> dict:
    """Descobre e ingere os editais novos de uma fonte. Devolve um resumo."""
    novos = descobrir_novos(
        source, ja_conhecido=lambda url: store.find_by_storage_path(url) is not None
    )
    logger.info("Descoberta: %d edital(is) novo(s)", len(novos))

    resultados = []
    for i, ref in enumerate(novos):
        if i > 0:
            time.sleep(_PAUSA_ENTRE_DOWNLOADS_SEGUNDOS)
        resultados.append(ingerir_edital(ref, store))

    sucesso = sum(1 for r in resultados if r.get("status") != "failed")
    falha = len(resultados) - sucesso
    resumo = {"descobertos": len(novos), "sucesso": sucesso, "falha": falha}
    logger.info("Ciclo concluído: %s", resumo)
    return resumo


def main() -> None:
    parser = argparse.ArgumentParser(description="Ingestão automática de editais do IFSC")
    parser.add_argument(
        "--loop",
        action="store_true",
        help="Roda continuamente, a cada settings.auto_ingest_ciclo_segundos",
    )
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")

    store = WeaviateStore(raw_collection_name())
    store.get_or_create_collection()
    try:
        while True:
            executar_ciclo(fonte_padrao(), store)
            if not args.loop:
                break
            time.sleep(settings.auto_ingest_ciclo_segundos)
    finally:
        store.close()


if __name__ == "__main__":
    main()
