"""Cache semântico de perguntas do RAG.

Antes de rodar `hybrid_search` + geração (`retrieval/generate.py`),
verifica se a pergunta atual é semanticamente equivalente a alguma
pergunta recente já respondida -- embedding da pergunta (mesmo
embedder do retrieval, Voyage) comparado por similaridade de cosseno
contra um scan linear das últimas `settings.rag_cache_max_itens`
perguntas cacheadas no Redis. Sem índice vetorial: proporcional ao
volume de um hackathon, não a produção em escala.

Cache por CONTEÚDO da pergunta, não por usuário -- duas pessoas
perguntando a mesma coisa batem no mesmo item.

Falha aberta: se o Redis (ou o embedder) estiver indisponível, `buscar`
devolve `None` e `salvar` não faz nada -- o RAG segue seu fluxo normal
sem cache, nunca quebra a resposta por causa disso.
"""

from __future__ import annotations

import json
import logging
import math
import os
import uuid

import redis
from redis.exceptions import RedisError

from config.settings import settings
from ingestion.embeddings import VoyageEmbedding

logger = logging.getLogger(__name__)

_CHAVE_LISTA = "rag_cache:perguntas"
_PREFIXO_ITEM = "rag_cache:item:"

_redis: redis.Redis | None = None
_embedder: VoyageEmbedding | None = None


def _get_redis() -> redis.Redis:
    """Cria a conexão com o Redis uma única vez e reaproveita.

    Cliente síncrono de propósito -- `retrieval.generate.answer` (quem
    chama este módulo) é síncrono, ao contrário de `channels/session.py`
    (que usa `redis.asyncio` porque o canal Telegram é assíncrono).
    """
    global _redis
    if _redis is None:
        url = os.environ.get("REDIS_URL", "redis://localhost:6380")
        _redis = redis.Redis.from_url(url, decode_responses=True)
    return _redis


def _get_embedder() -> VoyageEmbedding:
    global _embedder
    if _embedder is None:
        _embedder = VoyageEmbedding(collection_name="rag_semantic_cache")
    return _embedder


def _cosseno(a: list[float], b: list[float]) -> float:
    produto = sum(x * y for x, y in zip(a, b))
    norma_a = math.sqrt(sum(x * x for x in a))
    norma_b = math.sqrt(sum(y * y for y in b))
    if norma_a == 0 or norma_b == 0:
        return 0.0
    return produto / (norma_a * norma_b)


def buscar(pergunta: str) -> dict | None:
    """Devolve a resposta cacheada pra uma pergunta semanticamente
    equivalente já respondida, ou `None` (sem hit, cache desligado, ou
    Redis/embedder indisponível -- trate `None` sempre como "segue o
    fluxo normal").
    """
    if not settings.rag_cache_habilitado:
        return None
    try:
        r = _get_redis()
        chaves = r.lrange(_CHAVE_LISTA, 0, settings.rag_cache_max_itens - 1)
        if not chaves:
            return None

        vetor_pergunta = _get_embedder().Vectorize_documents([pergunta])[0]

        melhor_item: dict | None = None
        melhor_similaridade = -1.0
        for chave in chaves:
            bruto = r.get(_PREFIXO_ITEM + chave)
            if bruto is None:  # expirou ou foi removido -- pula
                continue
            item = json.loads(bruto)
            similaridade = _cosseno(vetor_pergunta, item["embedding"])
            if similaridade > melhor_similaridade:
                melhor_similaridade = similaridade
                melhor_item = item

        if melhor_item is None or melhor_similaridade < settings.rag_cache_limiar_similaridade:
            return None

        logger.info("Cache semântico: hit (similaridade=%.3f)", melhor_similaridade)
        return {
            "answer": melhor_item["resposta"],
            "sources": melhor_item["fontes"],
            "recusa": melhor_item["recusa"],
        }
    except RedisError as exc:
        logger.error("Cache semântico indisponível (Redis), pulando: %s", type(exc).__name__)
        return None
    except Exception as exc:
        # Nunca deixa uma falha de cache (embedder, JSON malformado etc)
        # quebrar a resposta -- só desativa o cache pra esta chamada.
        logger.error("Falha ao consultar cache semântico, pulando (%s)", type(exc).__name__)
        return None


def salvar(pergunta: str, resposta: str, fontes: list[str], recusa: bool) -> None:
    """Grava a pergunta respondida no cache.

    TTL mais curto pra recusa (`rag_cache_ttl_recusa_segundos`) que pra
    resposta com base real (`rag_cache_ttl_segundos`) -- o crawler roda
    periodicamente, então uma recusa de ontem pode não valer mais hoje.
    Mantém só as `rag_cache_max_itens` mais recentes, descartando as
    mais antigas (chave da lista E o item em si).
    """
    if not settings.rag_cache_habilitado:
        return
    try:
        vetor_pergunta = _get_embedder().Vectorize_documents([pergunta])[0]
        chave_item = uuid.uuid4().hex
        item = {
            "pergunta": pergunta,
            "embedding": vetor_pergunta,
            "resposta": resposta,
            "fontes": fontes,
            "recusa": recusa,
        }
        ttl = settings.rag_cache_ttl_recusa_segundos if recusa else settings.rag_cache_ttl_segundos

        r = _get_redis()
        r.set(_PREFIXO_ITEM + chave_item, json.dumps(item), ex=ttl)
        r.lpush(_CHAVE_LISTA, chave_item)

        excedentes = r.lrange(_CHAVE_LISTA, settings.rag_cache_max_itens, -1)
        if excedentes:
            r.ltrim(_CHAVE_LISTA, 0, settings.rag_cache_max_itens - 1)
            r.delete(*(_PREFIXO_ITEM + c for c in excedentes))
    except RedisError as exc:
        logger.error("Cache semântico indisponível (Redis) ao salvar, pulando: %s", type(exc).__name__)
    except Exception as exc:
        logger.error("Falha ao salvar no cache semântico, pulando (%s)", type(exc).__name__)
