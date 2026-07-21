"""
Testes puros do rate limit e da deduplicação -- não tocam Redis real.
Um Redis falso em memória (FakeRedis) simula incr/expire/set(nx),
e um que levanta erro confirma a decisão de falha aberta.
"""

import asyncio

import pytest

from channels import rate_limit


class FakeRedis:
    """Redis mínimo em memória, só com o que o rate_limit usa."""

    def __init__(self):
        self.contadores = {}
        self.chaves_set = set()

    async def incr(self, chave):
        self.contadores[chave] = self.contadores.get(chave, 0) + 1
        return self.contadores[chave]

    async def expire(self, chave, segundos):
        return True

    async def set(self, chave, valor, nx=False, ex=None):
        if nx and chave in self.chaves_set:
            return None  # já existe: não grava
        self.chaves_set.add(chave)
        return True


class RedisQuebrado:
    """Simula o Redis fora do ar: toda operação levanta RedisError."""

    async def incr(self, chave):
        from redis.exceptions import RedisError
        raise RedisError("conexão recusada")

    async def set(self, chave, valor, nx=False, ex=None):
        from redis.exceptions import RedisError
        raise RedisError("conexão recusada")


class FakeRedisComExpire(FakeRedis):
    """Redis fake que simula expiração de chaves por janela de tempo."""

    def __init__(self):
        super().__init__()
        self._expirations: dict[str, float] = {}
        self._now = 0.0

    async def incr(self, chave):
        expiry = self._expirations.get(chave)
        if expiry is not None and self._now >= expiry:
            self.contadores.pop(chave, None)
            self._expirations.pop(chave, None)
        return await super().incr(chave)

    async def expire(self, chave, segundos):
        self._expirations[chave] = self._now + segundos
        return True

    def advance(self, segundos):
        self._now += segundos


def test_dentro_do_limite_passa(monkeypatch):
    fake = FakeRedis()
    monkeypatch.setattr(rate_limit, "_get_redis", lambda: fake)
    monkeypatch.setattr(rate_limit.settings, "rate_limit_mensagens", 5)

    # as 5 primeiras devem passar
    for _ in range(5):
        assert asyncio.run(rate_limit.permitido("user1")) is True


def test_acima_do_limite_bloqueia(monkeypatch):
    fake = FakeRedis()
    monkeypatch.setattr(rate_limit, "_get_redis", lambda: fake)
    monkeypatch.setattr(rate_limit.settings, "rate_limit_mensagens", 5)

    for _ in range(5):
        asyncio.run(rate_limit.permitido("user1"))
    # a 6ª estoura o limite
    assert asyncio.run(rate_limit.permitido("user1")) is False


def test_usuarios_diferentes_nao_se_afetam(monkeypatch):
    fake = FakeRedis()
    monkeypatch.setattr(rate_limit, "_get_redis", lambda: fake)
    monkeypatch.setattr(rate_limit.settings, "rate_limit_mensagens", 2)

    asyncio.run(rate_limit.permitido("user1"))
    asyncio.run(rate_limit.permitido("user1"))
    # user1 estourou, mas user2 começa do zero
    assert asyncio.run(rate_limit.permitido("user1")) is False
    assert asyncio.run(rate_limit.permitido("user2")) is True


def test_falha_aberta_libera_quando_redis_cai(monkeypatch):
    monkeypatch.setattr(rate_limit, "_get_redis", lambda: RedisQuebrado())
    # Redis fora do ar: decisão consciente é LIBERAR
    assert asyncio.run(rate_limit.permitido("user1")) is True


def test_dedup_primeira_vez_passa_repeticao_bloqueia(monkeypatch):
    fake = FakeRedis()
    monkeypatch.setattr(rate_limit, "_get_redis", lambda: fake)

    # primeira vez: não é duplicada
    assert asyncio.run(rate_limit.eh_duplicada("user1", "oi")) is False
    # mesma mensagem de novo: é duplicada
    assert asyncio.run(rate_limit.eh_duplicada("user1", "oi")) is True


def test_dedup_mensagens_diferentes_nao_colidem(monkeypatch):
    fake = FakeRedis()
    monkeypatch.setattr(rate_limit, "_get_redis", lambda: fake)

    assert asyncio.run(rate_limit.eh_duplicada("user1", "oi")) is False
    assert asyncio.run(rate_limit.eh_duplicada("user1", "bom dia")) is False


def test_dedup_falha_aberta_quando_redis_cai(monkeypatch):
    monkeypatch.setattr(rate_limit, "_get_redis", lambda: RedisQuebrado())
    # Redis fora: trata como NÃO duplicada
    assert asyncio.run(rate_limit.eh_duplicada("user1", "oi")) is False


def test_permitido_reseta_apos_janela(monkeypatch):
    fake = FakeRedisComExpire()
    monkeypatch.setattr(rate_limit, "_get_redis", lambda: fake)
    monkeypatch.setattr(rate_limit.settings, "rate_limit_mensagens", 2)
    monkeypatch.setattr(rate_limit.settings, "rate_limit_janela_segundos", 10)

    assert asyncio.run(rate_limit.permitido("user1")) is True
    assert asyncio.run(rate_limit.permitido("user1")) is True
    assert asyncio.run(rate_limit.permitido("user1")) is False

    fake.advance(11)
    assert asyncio.run(rate_limit.permitido("user1")) is True