"""
Testes puros da parte de recusa em `retrieval/generate.py` -- so testam
`_fontes_relevantes`, que e pura, sem tocar Groq nem Weaviate. A propria
recusa nao e mais uma heuristica de texto: o modelo devolve um campo
"recusa" no JSON estruturado da resposta (`answer()`), e e esse booleano
que `_fontes_relevantes` recebe direto -- so um teste de integracao real
pode validar se o modelo classifica certo (ver tests/test_seguranca.py).
"""

from retrieval.generate import _MAX_SOURCES, _fontes_relevantes


class TestFontesRelevantes:
    def test_recusa_nunca_tem_fonte(self):
        hits = [{"file_name": "edital_01.pdf"}]
        assert _fontes_relevantes(hits, recusa=True) == []

    def test_resposta_ancorada_traz_fonte(self):
        hits = [{"file_name": "edital_01.pdf"}]
        assert _fontes_relevantes(hits, recusa=False) == ["edital_01.pdf"]

    def test_deduplica_preservando_ordem(self):
        hits = [
            {"file_name": "edital_01.pdf"},
            {"file_name": "edital_02.pdf"},
            {"file_name": "edital_01.pdf"},
        ]
        assert _fontes_relevantes(hits, recusa=False) == ["edital_01.pdf", "edital_02.pdf"]

    def test_respeita_limite_max_sources(self):
        hits = [{"file_name": f"edital_{i}.pdf"} for i in range(_MAX_SOURCES + 5)]
        assert len(_fontes_relevantes(hits, recusa=False)) == _MAX_SOURCES
