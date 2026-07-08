"""CLI de ingestão do Decifra.

Percorre a pasta de editais e roda o pipeline em cada documento.

Uso
---
    python run_ingest.py
    python run_ingest.py --data-dir data/editais
"""

from __future__ import annotations

import argparse
import logging

from config.settings import settings
from ingestion.local_source import iter_documents
from ingestion.pipeline import ingest_document, raw_collection_name
from ingestion.weaviate_store import WeaviateStore

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
logger = logging.getLogger("run_ingest")


def main() -> None:
    parser = argparse.ArgumentParser(description="Ingestão de editais no Decifra")
    parser.add_argument("--data-dir", default=settings.data_dir)
    args = parser.parse_args()

    store = WeaviateStore(raw_collection_name())
    store.get_or_create_collection()

    total = indexed = 0
    try:
        for file_name, content in iter_documents(args.data_dir):
            total += 1
            result = ingest_document(file_name, content, store)
            if result.get("status") not in {"skipped", "empty"}:
                indexed += 1
    finally:
        store.close()

    logger.info("Concluído: %d documentos, %d indexados nesta rodada", total, indexed)


if __name__ == "__main__":
    main()
