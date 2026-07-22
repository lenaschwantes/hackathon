"""Testes puros do cache semântico (`infra/semantic_cache.py`).

Redis fake (com TTL de verdade, via relógio real) e embedder fake
(vetores fixos por texto) -- sem tocar Redis nem Voyage de verdade.
"""

from __future__ import annotations

import time

import pytest
from redis.exceptions import ConnectionError as RedisConnectionError

import infra.semantic_cache as cache_module
from config.settings import settings

_PERGUNTA_ORIGINAL = "como faço pra me inscrever no técnico de informática?"
_PERGUNTA_PARAFRASE = "qual o processo de inscrição pro técnico de informática?"
_PERGUNTA_DIFERENTE = "quais os documentos exigidos pro curso de enfermagem?"

_VEC_ORIGINAL = [1.0, 0.0, 0.0]
_VEC_PARAFRASE = [0.99, 0.14, 0.0]  # cosseno ~0.99 com o original -- acima do limiar
_VEC_DIFERENTE = [0.0, 1.0, 0.0]  # ortogonal -- cosseno 0.0 com o original


class _FakeRedis:
    """Fake mínimo do `redis.Redis` síncrono -- só o que o cache usa.
    Ao contrário do fake de `tests/integration/test_seguranca.py`, este
    simula TTL de verdade (relógio real), porque o cache depende disso.
    """

    def __init__(self):
        self._store: dict[str, str] = {}
        self._expira_em: dict[str, float] = {}
        self._listas: dict[str, list[str]] = {}

    def _expirou(self, chave: str) -> bool:
        prazo = self._expira_em.get(chave)
        return prazo is not None and time.time() >= prazo

    def set(self, chave, valor, ex=None):
        self._store[chave] = valor
        if ex is not None:
            self._expira_em[chave] = time.time() + ex
        else:
            self._expira_em.pop(chave, None)
        return True

    def get(self, chave):
        if self._expirou(chave):
            self._store.pop(chave, None)
            self._expira_em.pop(chave, None)
            return None
        return self._store.get(chave)

    def lpush(self, chave, valor):
        self._listas.setdefault(chave, []).insert(0, valor)
        return len(self._listas[chave])

    def lrange(self, chave, start, end):
        lista = self._listas.get(chave, [])
        return lista[start:] if end == -1 else lista[start : end + 1]

    def ltrim(self, chave, start, end):
        lista = self._listas.get(chave, [])
        self._listas[chave] = lista[start:] if end == -1 else lista[start : end + 1]

    def delete(self, *chaves):
        for chave in chaves:
            self._store.pop(chave, None)
            self._expira_em.pop(chave, None)
        return len(chaves)


class _FakeRedisIndisponivel:
    """Simula Redis fora do ar -- toda operação levanta ConnectionError."""

    def __getattr__(self, _nome):
        def _levanta(*args, **kwargs):
            raise RedisConnectionError("Redis indisponível (simulado)")

        return _levanta


class _FakeEmbedder:
    def __init__(self, vetores: dict[str, list[float]]):
        self._vetores = vetores

    def Vectorize_documents(self, chunks: list[str]) -> list[list[float]]:
        return [self._vetores[c] for c in chunks]


@pytest.fixture(autouse=True)
def _reset_singletons(monkeypatch):
    # Os getters do módulo cacheiam num global -- garante que cada
    # teste comece sem conexão/embedder residual de um teste anterior.
    monkeypatch.setattr(cache_module, "_redis", None)
    monkeypatch.setattr(cache_module, "_embedder", None)
    monkeypatch.setattr(settings, "rag_cache_habilitado", True)
    monkeypatch.setattr(settings, "rag_cache_limiar_similaridade", 0.95)
    monkeypatch.setattr(settings, "rag_cache_ttl_segundos", 3600)
    monkeypatch.setattr(settings, "rag_cache_ttl_recusa_segundos", 1800)
    monkeypatch.setattr(settings, "rag_cache_max_itens", 80)


def _usa_fakes(monkeypatch, vetores: dict[str, list[float]]) -> _FakeRedis:
    fake_redis = _FakeRedis()
    monkeypatch.setattr(cache_module, "_get_redis", lambda: fake_redis)
    monkeypatch.setattr(cache_module, "_get_embedder", lambda: _FakeEmbedder(vetores))
    return fake_redis


class TestBuscarESalvar:
    def test_pergunta_identica_bate_no_cache(self, monkeypatch):
        _usa_fakes(monkeypatch, {_PERGUNTA_ORIGINAL: _VEC_ORIGINAL})

        cache_module.salvar(_PERGUNTA_ORIGINAL, "As inscrições vão até 20/08.", ["edital.pdf"], False)
        resultado = cache_module.buscar(_PERGUNTA_ORIGINAL)

        assert resultado == {
            "answer": "As inscrições vão até 20/08.",
            "sources": ["edital.pdf"],
            "recusa": False,
        }

    def test_parafrase_com_similaridade_alta_bate_no_cache(self, monkeypatch):
        _usa_fakes(
            monkeypatch,
            {_PERGUNTA_ORIGINAL: _VEC_ORIGINAL, _PERGUNTA_PARAFRASE: _VEC_PARAFRASE},
        )

        cache_module.salvar(_PERGUNTA_ORIGINAL, "As inscrições vão até 20/08.", ["edital.pdf"], False)
        resultado = cache_module.buscar(_PERGUNTA_PARAFRASE)

        assert resultado is not None
        assert resultado["answer"] == "As inscrições vão até 20/08."

    def test_pergunta_diferente_nao_bate(self, monkeypatch):
        _usa_fakes(
            monkeypatch,
            {_PERGUNTA_ORIGINAL: _VEC_ORIGINAL, _PERGUNTA_DIFERENTE: _VEC_DIFERENTE},
        )

        cache_module.salvar(_PERGUNTA_ORIGINAL, "As inscrições vão até 20/08.", ["edital.pdf"], False)
        resultado = cache_module.buscar(_PERGUNTA_DIFERENTE)

        assert resultado is None

    def test_cache_vazio_nao_bate_em_nada(self, monkeypatch):
        _usa_fakes(monkeypatch, {_PERGUNTA_ORIGINAL: _VEC_ORIGINAL})
        assert cache_module.buscar(_PERGUNTA_ORIGINAL) is None


class TestTTL:
    def test_recusa_expira_com_ttl_mais_curto(self, monkeypatch):
        monkeypatch.setattr(settings, "rag_cache_ttl_recusa_segundos", 0)
        _usa_fakes(monkeypatch, {_PERGUNTA_ORIGINAL: _VEC_ORIGINAL})

        cache_module.salvar(_PERGUNTA_ORIGINAL, "Não encontrei essa informação.", [], True)
        time.sleep(0.05)

        assert cache_module.buscar(_PERGUNTA_ORIGINAL) is None

    def test_resposta_com_base_expira_apos_ttl(self, monkeypatch):
        monkeypatch.setattr(settings, "rag_cache_ttl_segundos", 0)
        _usa_fakes(monkeypatch, {_PERGUNTA_ORIGINAL: _VEC_ORIGINAL})

        cache_module.salvar(_PERGUNTA_ORIGINAL, "As inscrições vão até 20/08.", ["edital.pdf"], False)
        time.sleep(0.05)

        assert cache_module.buscar(_PERGUNTA_ORIGINAL) is None

    def test_dentro_do_ttl_continua_valido(self, monkeypatch):
        monkeypatch.setattr(settings, "rag_cache_ttl_segundos", 3600)
        _usa_fakes(monkeypatch, {_PERGUNTA_ORIGINAL: _VEC_ORIGINAL})

        cache_module.salvar(_PERGUNTA_ORIGINAL, "As inscrições vão até 20/08.", ["edital.pdf"], False)
        assert cache_module.buscar(_PERGUNTA_ORIGINAL) is not None


class TestLimiteDeItens:
    def test_descarta_os_mais_antigos_alem_do_limite(self, monkeypatch):
        monkeypatch.setattr(settings, "rag_cache_max_itens", 2)
        perguntas = ["pergunta a", "pergunta b", "pergunta c"]
        vetores = {
            "pergunta a": [1.0, 0.0, 0.0],
            "pergunta b": [0.0, 1.0, 0.0],
            "pergunta c": [0.0, 0.0, 1.0],
        }
        fake_redis = _usa_fakes(monkeypatch, vetores)

        for p in perguntas:
            cache_module.salvar(p, f"resposta pra {p}", [], False)

        # só "b" e "c" (as 2 mais recentes) devem sobrar na lista.
        assert fake_redis.lrange(cache_module._CHAVE_LISTA, 0, -1).__len__() == 2

        # "a" (a mais antiga) foi descartada -- buscar por ela de novo
        # não pode bater em si mesma via cache (o item sumiu).
        assert cache_module.buscar("pergunta a") is None
        assert cache_module.buscar("pergunta c") is not None


class TestFalhaAberta:
    def test_redis_indisponivel_buscar_devolve_none_sem_quebrar(self, monkeypatch):
        monkeypatch.setattr(cache_module, "_get_redis", lambda: _FakeRedisIndisponivel())
        monkeypatch.setattr(cache_module, "_get_embedder", lambda: _FakeEmbedder({}))

        assert cache_module.buscar("qualquer pergunta") is None

    def test_redis_indisponivel_salvar_nao_levanta(self, monkeypatch):
        monkeypatch.setattr(cache_module, "_get_redis", lambda: _FakeRedisIndisponivel())
        monkeypatch.setattr(
            cache_module, "_get_embedder", lambda: _FakeEmbedder({"pergunta": [1.0, 0.0, 0.0]})
        )

        cache_module.salvar("pergunta", "resposta", [], False)  # não deve levantar

    def test_cache_desabilitado_buscar_devolve_none_sem_tocar_redis(self, monkeypatch):
        monkeypatch.setattr(settings, "rag_cache_habilitado", False)

        def _explode():
            raise AssertionError("não deveria tocar o Redis com o cache desligado")

        monkeypatch.setattr(cache_module, "_get_redis", _explode)

        assert cache_module.buscar("qualquer pergunta") is None
