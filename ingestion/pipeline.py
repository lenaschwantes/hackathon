"""Pipeline de ingestão do Decifra.

Fluxo: bytes do edital -> extração -> limpeza -> deduplicação por hash ->
grava o texto cru (proveniência) -> chunk -> embed (Voyage) -> insere os
chunks vetorizados no Weaviate.
"""

from __future__ import annotations

import logging

from config.settings import settings
from ingestion.clean import clean_text
from ingestion.embeddings import VoyageEmbedding
from ingestion.extract import extract_text
from ingestion.simple_chunk import chunk_text
from ingestion.weaviate_store import WeaviateStore
from utils.hashing import sha256_bytes

logger = logging.getLogger(__name__)


def raw_collection_name() -> str:
    """Nome da coleção (ex.: ``Editais_raw``)."""
    return f"{settings.default_collection}{settings.raw_collection_suffix}"


def ingest_document(file_name: str, content: bytes, store: WeaviateStore) -> dict:
    """Ingere um único documento: texto cru + chunks vetorizados.

    Parameters
    ----------
    file_name : str
        Nome do arquivo de origem.
    content : bytes
        Conteúdo binário do documento.
    store : WeaviateStore
        Store já conectada à coleção.

    Returns
    -------
    dict
        Resultado do upsert, ou dict de status quando pulado.
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

    chunks = chunk_text(text)
    if chunks:
        embedder = VoyageEmbedding(collection_name=store.collection_name)
        vectors = embedder.Vectorize_documents(chunks)
        store.insert_chunks(
            file_name=file_name,
            file_hash=file_hash,
            chunks=chunks,
            vectors=vectors,
        )
        logger.info("Indexados %d chunks: %s", len(chunks), file_name)

    return result
