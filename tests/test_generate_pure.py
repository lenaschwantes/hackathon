"""
Testes puros da parte de recusa em `retrieval/generate.py` -- so
testam `_eh_recusa` e `_fontes_relevantes`, que sao heuristicas de
string, sem tocar Groq nem Weaviate. E aqui que mora o criterio de
fidelidade: a fonte so aparece quando a resposta realmente se ancorou
nos trechos, nunca numa recusa.
"""

from retrieval.generate import _MAX_SOURCES, _eh_recusa, _fontes_relevantes


class TestEhRecusa:
    def test_reconhece_frase_de_recusa(self):
        assert _eh_recusa("Não encontrei essa informação nos editais que tenho aqui.")

    def test_reconhece_recusa_case_insensitive(self):
        assert _eh_recusa("NÃO ENCONTREI ESSA INFORMAÇÃO no acervo.")

    def test_resposta_ancorada_nao_e_recusa(self):
        texto = "A cota é reservada para quem cursou o ensino médio em escola pública."
        assert not _eh_recusa(texto)

    def test_texto_vazio_ou_none_nao_e_recusa(self):
        assert not _eh_recusa("")
        assert not _eh_recusa(None)


class TestFontesRelevantes:
    def test_recusa_nunca_tem_fonte(self):
        hits = [{"file_name": "edital_01.pdf"}]
        texto = "Não encontrei essa informação nos editais que tenho aqui."
        assert _fontes_relevantes(hits, texto) == []

    def test_resposta_ancorada_traz_fonte(self):
        hits = [{"file_name": "edital_01.pdf"}]
        texto = "A inscrição vai até 20/07."
        assert _fontes_relevantes(hits, texto) == ["edital_01.pdf"]

    def test_deduplica_preservando_ordem(self):
        hits = [
            {"file_name": "edital_01.pdf"},
            {"file_name": "edital_02.pdf"},
            {"file_name": "edital_01.pdf"},
        ]
        texto = "A inscrição vai até 20/07."
        assert _fontes_relevantes(hits, texto) == ["edital_01.pdf", "edital_02.pdf"]

    def test_respeita_limite_max_sources(self):
        hits = [{"file_name": f"edital_{i}.pdf"} for i in range(_MAX_SOURCES + 5)]
        texto = "A inscrição vai até 20/07."
        assert len(_fontes_relevantes(hits, texto)) == _MAX_SOURCES
