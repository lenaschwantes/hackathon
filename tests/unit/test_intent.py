"""
Testes puros do roteamento de intenção pré-RAG -- não tocam Anthropic.
A camada 2 (LLM) é isolada em `_chamar_llm_classificador`, mesmo padrão
de `tests/test_recommendation.py` pra `quer_nova_recomendacao`.
"""

from dialogue import intent
from dialogue.intent import _bate_keyword, precisa_busca


class TestBateKeyword:
    def test_reconhece_palavra_chave(self):
        assert _bate_keyword("Qual o prazo do edital?") is True

    def test_reconhece_sem_acento_e_caixa(self):
        assert _bate_keyword("QUANDO FECHA A INSCRICAO") is True

    def test_nao_reconhece_papo_informal(self):
        assert _bate_keyword("oi, tudo bem?") is False

    def test_nao_reconhece_substring_dentro_de_outra_palavra(self):
        # "vaga" não deveria bater dentro de "enxovaga" (palavra inventada,
        # só pra garantir que a regex respeita fronteira de palavra).
        assert _bate_keyword("isso e uma enxovagante bobagem") is False


class TestPrecisaBusca:
    def test_keyword_decide_sem_chamar_llm(self, monkeypatch):
        def _chamar_llm_nao_deveria_ser_chamado(texto):
            raise AssertionError("_chamar_llm_classificador não deveria ser chamado quando keyword bate")

        monkeypatch.setattr(intent, "_chamar_llm_classificador", _chamar_llm_nao_deveria_ser_chamado)

        assert precisa_busca("Qual o prazo de inscrição?") is True

    def test_sem_keyword_usa_classificador_true(self, monkeypatch):
        monkeypatch.setattr(intent, "_chamar_llm_classificador", lambda texto: {"precisa_busca": True})
        assert precisa_busca("quem tem direito a cota?") is True

    def test_sem_keyword_usa_classificador_false(self, monkeypatch):
        monkeypatch.setattr(intent, "_chamar_llm_classificador", lambda texto: {"precisa_busca": False})
        assert precisa_busca("oi, tudo bem?") is False

    def test_falha_no_classificador_assume_que_precisa_buscar(self, monkeypatch):
        """
        Ao contrário de `quer_nova_recomendacao` (assume False na
        dúvida), aqui a falha assume True -- é pior deixar de responder
        uma pergunta real sobre edital do que rodar uma busca à toa.
        """
        def _chamar_llm_com_erro(texto):
            raise RuntimeError("Anthropic indisponível")

        monkeypatch.setattr(intent, "_chamar_llm_classificador", _chamar_llm_com_erro)
        assert precisa_busca("qualquer coisa") is True
