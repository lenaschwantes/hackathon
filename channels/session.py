"""
Estado da conversa de cada usuário, guardado no Redis.

Cada usuário tem uma "ficha" própria (chave `sessao:{user_id}`),
guardada como JSON. Essa ficha expira sozinha depois de 24h sem uso
(TTL), pra não acumular sessão de gente que abandonou a conversa.

Schema fixo, não inventar campo além destes três:
- perfil: dict, comeca vazio, preenchido conforme a conversa avanca
- fase_dialogo: string, em que ponto do dialogo a pessoa esta
- historico: lista das ultimas mensagens (no maximo 10)
"""

import json
import os
from redis.asyncio import Redis

TTL_SEGUNDOS = 60 * 60 * 24  # 24h
MAX_HISTORICO = 10

_redis: Redis | None = None


def _get_redis() -> Redis:
    """
    Cria a conexão com o Redis uma única vez e reaproveita.
    A URL vem do .env; se não tiver, usa o padrão do docker-compose.
    """
    global _redis
    if _redis is None:
        url = os.environ.get("REDIS_URL", "redis://localhost:6380")
        _redis = Redis.from_url(url, decode_responses=True)
    return _redis


def _chave(user_id: str) -> str:
    return f"sessao:{user_id}"


def _sessao_vazia() -> dict:
    return {"perfil": {}, "fase_dialogo": "inicio", "historico": []}


async def carregar_sessao(user_id: str) -> dict:
    """
    Busca a sessão do usuário no Redis. Se ele nunca conversou antes
    (ou a sessão expirou), devolve uma sessão vazia — nunca None,
    pra quem chamar essa função não precisar tratar esse caso.
    """
    r = _get_redis()
    bruto = await r.get(_chave(user_id))
    if bruto is None:
        return _sessao_vazia()
    return json.loads(bruto)


async def salvar_sessao(user_id: str, sessao: dict) -> None:
    """
    Salva a sessão de volta no Redis, truncando o histórico pras
    últimas MAX_HISTORICO mensagens, e renovando o TTL de 24h
    a cada mensagem (conversa ativa nunca expira no meio do uso).
    """
    sessao["historico"] = sessao["historico"][-MAX_HISTORICO:]
    r = _get_redis()
    await r.set(_chave(user_id), json.dumps(sessao), ex=TTL_SEGUNDOS)