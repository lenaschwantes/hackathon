"""Retrieval híbrido sobre a coleção de editais.

Combina busca vetorial com BM25 (busca híbrida do Weaviate), controlada
por ``alpha`` (0 = só BM25, 1 = só vetor). Os defaults reproduzem o
projeto original: ``alpha=0.6``, ``k=12``.
"""

from __future__ import annotations

import weaviate
from weaviate.classes.query import MetadataQuery

from config.settings import settings


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
        Nome da coleção. Default: ``settings.default_collection``.
    k : int, optional
        Número de trechos a retornar. Default: ``settings.search_k``.
    alpha : float, optional
        Peso vetor vs BM25. Default: ``settings.search_alpha``.

    Returns
    -------
    list[dict]
        Trechos com ``text``, ``score`` e as propriedades de proveniência
        (nome do edital, etc.) para citar a fonte.
    """
    collection = collection or settings.default_collection
    k = k or settings.search_k
    alpha = settings.search_alpha if alpha is None else alpha

    client = weaviate.connect_to_local(
        host=settings.weaviate_http_url.split("//")[-1].split(":")[0]
    )
    try:
        coll = client.collections.get(collection)
        response = coll.query.hybrid(
            query=query,
            alpha=alpha,
            limit=k,
            return_metadata=MetadataQuery(score=True),
        )
        return [
            {
                "text": o.properties.get("content", ""),
                "file_name": o.properties.get("file_name", ""),
                "score": o.metadata.score,
                "properties": o.properties,
            }
            for o in response.objects
        ]
    finally:
        client.close()
