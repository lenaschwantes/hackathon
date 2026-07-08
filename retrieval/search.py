"""Retrieval híbrido sobre a coleção de editais.

Como o embedding é externo (Voyage, ``vectorizer=none``), a busca híbrida
precisa receber o vetor da pergunta explicitamente e mirar o vetor
nomeado ``content_vector``. Defaults do projeto original: alpha=0.6, k=12.
"""

from __future__ import annotations

import weaviate
from weaviate.classes.query import MetadataQuery

from config.settings import settings
from ingestion.embeddings import VoyageEmbedding


def _host() -> str:
    return settings.weaviate_http_url.split("//")[-1].split(":")[0]


def hybrid_search(
    query: str,
    collection: str | None = None,
    k: int | None = None,
    alpha: float | None = None,
) -> list[dict]:
    """Recupera os trechos mais relevantes para uma pergunta.

    Parameters
    ----------
    query : str
        Pergunta do cidadão, em linguagem natural.
    collection : str, optional
        Coleção. Default: ``settings.default_collection`` + sufixo raw.
    k : int, optional
        Número de trechos. Default: ``settings.search_k``.
    alpha : float, optional
        Peso vetor vs BM25. Default: ``settings.search_alpha``.

    Returns
    -------
    list[dict]
        Trechos com ``text``, ``file_name`` e ``score``.
    """
    collection = collection or (
        f"{settings.default_collection}{settings.raw_collection_suffix}"
    )
    k = k or settings.search_k
    alpha = settings.search_alpha if alpha is None else alpha

    embedder = VoyageEmbedding(collection_name=collection)
    query_vector = embedder.Vectorize_documents([query])[0]

    client = weaviate.connect_to_local(host=_host())
    try:
        coll = client.collections.get(collection)
        response = coll.query.hybrid(
            query=query,
            vector=query_vector,
            target_vector="content_vector",
            alpha=alpha,
            limit=k,
            return_metadata=MetadataQuery(score=True),
        )
        return [
            {
                "text": o.properties.get("content", ""),
                "file_name": o.properties.get("file_name", ""),
                "score": o.metadata.score,
            }
            for o in response.objects
        ]
    finally:
        client.close()
