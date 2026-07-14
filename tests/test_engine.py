"""
Testes puros da orquestracao em `channels/engine.py::responder()` --
nao tocam Groq, Weaviate nem Redis. Todas as chamadas de LLM/RAG/
recomendacao sao isoladas em funcoes proprias (`extrair_perfil`,
`_gerar_pergunta_coleta`, `gerar_recomendacao`, `answer`), entao os
testes monkeypatcham essas referencias dentro do modulo `engine`.
"""

from channels import engine
from channels.engine import _MENSAGEM_FALLBACK, responder
from dialogue.profile import Perfil


def _sessao(perfil: dict, fase: str = "coletando") -> dict:
    return {"perfil": perfil, "fase_dialogo": fase, "historico": []}


class TestPerfilCompletaNesteTurno:
    def test_devolve_recomendacao_e_nao_chama_rag(self, monkeypatch):
        perfil_incompleto = {
            "cidade": "Blumenau",
            "escolaridade": "ensino medio completo",
            "interesse": None,
            "modalidade": None,
        }
        perfil_completo = Perfil(**{**perfil_incompleto, "interesse": "tecnologia"})

        monkeypatch.setattr(engine, "extrair_perfil", lambda texto, perfil_atual: perfil_completo)
        monkeypatch.setattr(engine, "gerar_recomendacao", lambda perfil: "Achei um curso pra você!")

        def _answer_nao_deveria_ser_chamado(texto):
            raise AssertionError("answer() não deveria ser chamado quando o perfil acaba de completar")

        monkeypatch.setattr(engine, "answer", _answer_nao_deveria_ser_chamado)

        sessao = _sessao(perfil_incompleto)
        resposta = responder("user-1", "quero tecnologia", sessao)

        assert resposta == "Achei um curso pra você!"
        assert sessao["fase_dialogo"] == "completo"
        assert sessao["perfil"]["interesse"] == "tecnologia"

    def test_falha_na_recomendacao_cai_no_fallback(self, monkeypatch):
        perfil_incompleto = {
            "cidade": "Blumenau",
            "escolaridade": "ensino medio completo",
            "interesse": None,
            "modalidade": None,
        }
        perfil_completo = Perfil(**{**perfil_incompleto, "interesse": "tecnologia"})

        monkeypatch.setattr(engine, "extrair_perfil", lambda texto, perfil_atual: perfil_completo)

        def _gerar_recomendacao_com_erro(perfil):
            raise RuntimeError("Groq indisponível")

        monkeypatch.setattr(engine, "gerar_recomendacao", _gerar_recomendacao_com_erro)

        sessao = _sessao(perfil_incompleto)
        resposta = responder("user-1", "quero tecnologia", sessao)

        assert resposta == _MENSAGEM_FALLBACK


class TestPerfilJaCompletoAntesDoTurno:
    def test_cai_direto_no_rag_sem_recomendar_de_novo(self, monkeypatch):
        perfil_completo = {
            "cidade": "Blumenau",
            "escolaridade": "ensino medio completo",
            "interesse": "tecnologia",
            "modalidade": None,
        }

        def _extrair_perfil_nao_deveria_ser_chamado(texto, perfil_atual):
            raise AssertionError("extrair_perfil() não deveria ser chamado com perfil já completo")

        def _gerar_recomendacao_nao_deveria_ser_chamado(perfil):
            raise AssertionError("gerar_recomendacao() não deveria ser chamado de novo")

        monkeypatch.setattr(engine, "extrair_perfil", _extrair_perfil_nao_deveria_ser_chamado)
        monkeypatch.setattr(engine, "gerar_recomendacao", _gerar_recomendacao_nao_deveria_ser_chamado)
        monkeypatch.setattr(
            engine, "answer", lambda texto: {"answer": "A cota é...", "sources": ["edital.pdf"]}
        )

        sessao = _sessao(perfil_completo, fase="completo")
        resposta = responder("user-1", "o que é cota?", sessao)

        assert "A cota é..." in resposta
