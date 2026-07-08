"""Pipeline de ingestão do Decifra.

Fluxo: bytes do edital -> extração de texto -> limpeza -> deduplicação
por hash -> gravação na camada raw do Weaviate.

O passo de chunk + embed vive no serviço original de RAG e ainda não foi
portado para cá. Veja o TODO ao final: é onde o time pluga
``ingestion.chunking`` + ``ingestion.embeddings`` na Fase 0.
"""

from __future__ import annotations

import logging

from config.settings import settings
from ingestion.clean import clean_text
from ingestion.extract import extract_text
from ingestion.weaviate_store import WeaviateStore
from utils.hashing import sha256_bytes

logger = logging.getLogger(__name__)


def raw_collection_name() -> str:
    """Nome da coleção raw (ex.: ``Editais_raw``)."""
    return f"{settings.default_collection}{settings.raw_collection_suffix}"


def ingest_document(file_name: str, content: bytes, store: WeaviateStore) -> dict:
    """Ingere um único documento na camada raw.

    Parameters
    ----------
    file_name : str
        Nome do arquivo de origem.
    content : bytes
        Conteúdo binário do documento.
    store : WeaviateStore
        Store já conectada à coleção raw.

    Returns
    -------
    dict
        Resultado do upsert, ou um dict de status quando pulado.
    """
    file_hash = sha256_bytes(content)

    if store.find_by_file_hash(file_hash):
        logger.info("Documento já indexado, pulando: %s", file_name)
        return {"status": "skipped", "file_name": file_name, "file_hash": file_hash}

    text, meta = extract_text(file_name, content)
    text = clean_text(text)

    if len(text) < settings.min_extracted_chars:
        logger.warning("Texto extraído muito curto, pulando: %s", file_name)
        return {"status": "empty", "file_name": file_name}

    result = store.upsert_document(
        content=text,
        file_name=file_name,
        file_hash=file_hash,
        bucket="local",
        storage_path=file_name,
        content_type=meta.get("content_type"),
        text_chars=len(text),
        extractor=meta.get("extractor", "unknown"),
        source_format=meta.get("source_format"),
        converted_from=meta.get("converted_from"),
    )
    logger.info("Documento indexado: %s (%d chars)", file_name, len(text))

    # TODO (Fase 0): chunk + embed + insert_chunks.
    #   from ingestion.chunking import CustomHybridChunker
    #   from ingestion.embeddings import VoyageEmbedding
    #   chunks = CustomHybridChunker(...).chunk(dl_doc)   # precisa do DoclingDocument
    #   vectors = VoyageEmbedding().embed([c.text for c in chunks])
    #   store.insert_chunks(file_hash=file_hash, chunks=chunks, vectors=vectors)
    return result
