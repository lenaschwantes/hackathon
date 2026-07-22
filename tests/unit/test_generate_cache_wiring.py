"""Testa só a integração do cache semântico em `retrieval.generate.answer`
-- que ele é consultado ANTES de `hybrid_search`/LLM, e que um hit evita
os dois de vez. O comportamento do cache em si (similaridade, TTL,
limite de itens) é coberto em `tests/unit/test_semantic_cache.py`.
"""

from __future__ import annotations

import retrieval.generate as generate_module


class _ClientExplode:
    """Qualquer uso do client Anthropic aqui é falha de teste -- cache
    hit deveria ter evitado a geração por completo."""

    def __getattr__(self, _nome):
        raise AssertionError("LLM não deveria ter sido chamado com cache hit")


def test_cache_hit_evita_hybrid_search_e_llm(monkeypatch):
    resposta_cacheada = {
        "answer": "As inscrições vão até 20/08.",
        "sources": ["edital.pdf"],
        "recusa": False,
    }
    monkeypatch.setattr(generate_module.semantic_cache, "buscar", lambda pergunta: resposta_cacheada)

    def _hybrid_search_explode(*args, **kwargs):
        raise AssertionError("hybrid_search não deveria ter sido chamado com cache hit")

    monkeypatch.setattr(generate_module, "hybrid_search", _hybrid_search_explode)
    monkeypatch.setattr(generate_module.anthropic, "Anthropic", lambda: _ClientExplode())

    resultado = generate_module.answer("como me inscrevo no técnico de informática?")

    assert resultado == resposta_cacheada


def test_cache_miss_segue_fluxo_normal_e_grava_no_final(monkeypatch):
    monkeypatch.setattr(generate_module.semantic_cache, "buscar", lambda pergunta: None)

    salvos = []
    monkeypatch.setattr(
        generate_module.semantic_cache,
        "salvar",
        lambda pergunta, resposta, fontes, recusa: salvos.append(
            (pergunta, resposta, fontes, recusa)
        ),
    )
    monkeypatch.setattr(generate_module, "hybrid_search", lambda question, k=None: [])

    resultado = generate_module.answer("pergunta sem hit no cache")

    assert resultado["recusa"] is True
    assert salvos == [("pergunta sem hit no cache", generate_module._SEM_BASE, [], True)]
