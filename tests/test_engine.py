"""
Testes puros da orquestracao em `channels/engine.py::responder()` --
nao tocam Anthropic, Weaviate nem Redis. Todas as chamadas de LLM/RAG/
recomendacao sao isoladas em funcoes proprias (`extrair_perfil`,
`_gerar_pergunta_coleta`, `gerar_recomendacao`, `quer_nova_recomendacao`,
`answer`), entao os testes monkeypatcham essas referencias dentro do
modulo `engine`.
"""

from channels import engine
from channels.engine import _MENSAGEM_FALLBACK, responder
from dialogue.profile import Perfil


def _sessao(perfil: dict, fase: str = "coletando") -> dict:
    return {"perfil": perfil, "fase_dialogo": fase, "historico": []}


_PERFIL_INCOMPLETO = {
    "cidade": "Blumenau",
    "escolaridade": "ensino medio completo",
    "interesse": None,
    "nivel": None,
    "modalidade": None,
}

_PERFIL_COMPLETO = {
    "cidade": "Blumenau",
    "escolaridade": "ensino medio completo",
    "interesse": "tecnologia",
    "nivel": "tecnico integrado",
    "modalidade": None,
}


class TestPerfilCompletaNesteTurno:
    def test_devolve_recomendacao_e_nao_chama_rag(self, monkeypatch):
        perfil_completo = Perfil(**{**_PERFIL_INCOMPLETO, "interesse": "tecnologia", "nivel": "tecnico integrado"})

        monkeypatch.setattr(
            engine, "extrair_perfil", lambda texto, perfil_atual, historico=None: perfil_completo
        )
        monkeypatch.setattr(engine, "gerar_recomendacao", lambda perfil: "Achei um curso pra você!")

        def _answer_nao_deveria_ser_chamado(texto):
            raise AssertionError("answer() não deveria ser chamado quando o perfil acaba de completar")

        monkeypatch.setattr(engine, "answer", _answer_nao_deveria_ser_chamado)

        sessao = _sessao(dict(_PERFIL_INCOMPLETO))
        resposta = responder("user-1", "quero tecnologia", sessao)

        assert resposta == "Achei um curso pra você!"
        assert sessao["fase_dialogo"] == "completo"
        assert sessao["perfil"]["interesse"] == "tecnologia"

    def test_falha_na_recomendacao_cai_no_fallback(self, monkeypatch):
        perfil_completo = Perfil(**{**_PERFIL_INCOMPLETO, "interesse": "tecnologia", "nivel": "tecnico integrado"})

        monkeypatch.setattr(
            engine, "extrair_perfil", lambda texto, perfil_atual, historico=None: perfil_completo
        )

        def _gerar_recomendacao_com_erro(perfil):
            raise RuntimeError("Anthropic indisponível")

        monkeypatch.setattr(engine, "gerar_recomendacao", _gerar_recomendacao_com_erro)

        sessao = _sessao(dict(_PERFIL_INCOMPLETO))
        resposta = responder("user-1", "quero tecnologia", sessao)

        assert resposta == _MENSAGEM_FALLBACK


class TestPerfilJaCompletoAntesDoTurno:
    def _mocks_que_nao_deveriam_rodar(self, monkeypatch):
        def _extrair_perfil_nao_deveria_ser_chamado(texto, perfil_atual, historico=None):
            raise AssertionError("extrair_perfil() não deveria ser chamado com perfil já completo")

        monkeypatch.setattr(engine, "extrair_perfil", _extrair_perfil_nao_deveria_ser_chamado)

    def test_pergunta_normal_cai_no_rag_sem_recomendar_de_novo(self, monkeypatch):
        self._mocks_que_nao_deveriam_rodar(monkeypatch)

        def _gerar_recomendacao_nao_deveria_ser_chamado(perfil):
            raise AssertionError("gerar_recomendacao() não deveria ser chamado de novo")

        monkeypatch.setattr(engine, "gerar_recomendacao", _gerar_recomendacao_nao_deveria_ser_chamado)
        monkeypatch.setattr(engine, "quer_nova_recomendacao", lambda texto: False)
        monkeypatch.setattr(
            engine, "answer", lambda texto: {"answer": "A cota é...", "sources": ["edital.pdf"]}
        )

        sessao = _sessao(dict(_PERFIL_COMPLETO), fase="completo")
        resposta = responder("user-1", "o que é cota?", sessao)

        assert "A cota é..." in resposta

    def test_pedido_explicito_gera_nova_recomendacao_sem_chamar_rag(self, monkeypatch):
        self._mocks_que_nao_deveriam_rodar(monkeypatch)

        def _answer_nao_deveria_ser_chamado(texto):
            raise AssertionError("answer() não deveria ser chamado quando a pessoa pede nova recomendação")

        monkeypatch.setattr(engine, "quer_nova_recomendacao", lambda texto: True)
        monkeypatch.setattr(engine, "gerar_recomendacao", lambda perfil: "Que tal essa outra opção?")
        monkeypatch.setattr(engine, "answer", _answer_nao_deveria_ser_chamado)

        sessao = _sessao(dict(_PERFIL_COMPLETO), fase="completo")
        resposta = responder("user-1", "mostra outra opção", sessao)

        assert resposta == "Que tal essa outra opção?"

    def test_falha_ao_gerar_nova_recomendacao_cai_no_fallback(self, monkeypatch):
        self._mocks_que_nao_deveriam_rodar(monkeypatch)

        def _gerar_recomendacao_com_erro(perfil):
            raise RuntimeError("Anthropic indisponível")

        monkeypatch.setattr(engine, "quer_nova_recomendacao", lambda texto: True)
        monkeypatch.setattr(engine, "gerar_recomendacao", _gerar_recomendacao_com_erro)

        sessao = _sessao(dict(_PERFIL_COMPLETO), fase="completo")
        resposta = responder("user-1", "mostra outra opção", sessao)

        assert resposta == _MENSAGEM_FALLBACK


class TestHistoricoNaExtracao:
    def test_historico_da_sessao_e_repassado_pra_extracao(self, monkeypatch):
        capturado = {}

        def fake_extrair_perfil(texto, perfil_atual, historico=None):
            capturado["historico"] = historico
            return Perfil(**perfil_atual)

        monkeypatch.setattr(engine, "extrair_perfil", fake_extrair_perfil)
        monkeypatch.setattr(engine, "_gerar_pergunta_coleta", lambda perfil: "Pergunta de coleta.")

        sessao = _sessao(dict(_PERFIL_INCOMPLETO))
        sessao["historico"] = [{"de": "usuario", "texto": "oi"}, {"de": "bot", "texto": "olá!"}]

        responder("user-1", "moro em Blumenau", sessao)

        assert capturado["historico"] == sessao["historico"]


class TestTetoDeMensagem:
    def test_mensagem_gigante_e_truncada_antes_de_qualquer_coisa(self, monkeypatch):
        capturado = {}

        def fake_extrair_perfil(texto, perfil_atual, historico=None):
            capturado["tamanho"] = len(texto)
            return Perfil(**perfil_atual)

        monkeypatch.setattr(engine, "extrair_perfil", fake_extrair_perfil)
        monkeypatch.setattr(engine, "_gerar_pergunta_coleta", lambda perfil: "Pergunta de coleta.")

        sessao = _sessao(dict(_PERFIL_INCOMPLETO))
        responder("user-1", "A" * 50_000, sessao)

        assert capturado["tamanho"] <= engine._MAX_CARACTERES_MENSAGEM
