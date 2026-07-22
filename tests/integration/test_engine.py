"""
Testes puros da orquestracao em `channels/engine.py::responder()` --
nao tocam Anthropic, Weaviate nem Redis. Todas as chamadas de LLM/RAG/
recomendacao sao isoladas em funcoes proprias (`extrair_perfil`,
`_gerar_pergunta_coleta`, `gerar_recomendacao`, `quer_nova_recomendacao`,
`answer`), entao os testes monkeypatcham essas referencias dentro do
modulo `engine`.
"""

import pytest

from channels import engine
from channels.engine import _MENSAGEM_FALLBACK, responder
from dialogue.profile import Perfil


@pytest.fixture(autouse=True)
def _sem_classificador_de_reinicio_real(monkeypatch):
    """
    Autouse: garante que nenhum teste deste arquivo bata na API real do
    Anthropic via `classificar_pedido_reinicio` quando a chave local for
    valida -- sem isso, qualquer teste que exercite o perfil ja
    completo sem mockar esse classificador explicitamente faria uma
    chamada de rede de verdade (lenta e sujeita a rate limit),
    violando a promessa do docstring deste arquivo. Testes que
    precisam testar o classificador de verdade (`TestReinicio`,
    `TestBotoesDeReinicio`) sobrescrevem isso normalmente com seu
    proprio `monkeypatch.setattr` dentro do corpo do teste.
    """
    monkeypatch.setattr(engine, "classificar_pedido_reinicio", lambda texto: "nenhum")


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
    def test_pergunta_normal_cai_no_rag_sem_recomendar_de_novo(self, monkeypatch):
        def _extrair_perfil_nao_deveria_ser_chamado(texto, perfil_atual, historico=None):
            raise AssertionError("extrair_perfil() não deveria ser chamado numa pergunta normal (RAG)")

        monkeypatch.setattr(engine, "extrair_perfil", _extrair_perfil_nao_deveria_ser_chamado)

        def _gerar_recomendacao_nao_deveria_ser_chamado(perfil):
            raise AssertionError("gerar_recomendacao() não deveria ser chamado de novo")

        monkeypatch.setattr(engine, "gerar_recomendacao", _gerar_recomendacao_nao_deveria_ser_chamado)
        monkeypatch.setattr(engine, "quer_nova_recomendacao", lambda texto: False)
        monkeypatch.setattr(engine, "precisa_busca", lambda texto: True)
        monkeypatch.setattr(
            engine, "answer", lambda texto: {"answer": "A cota é...", "sources": ["edital.pdf"]}
        )

        sessao = _sessao(dict(_PERFIL_COMPLETO), fase="completo")
        resposta = responder("user-1", "o que é cota?", sessao)

        assert "A cota é..." in resposta

    def test_papo_informal_nao_chama_rag(self, monkeypatch):
        def _extrair_perfil_nao_deveria_ser_chamado(texto, perfil_atual, historico=None):
            raise AssertionError("extrair_perfil() não deveria ser chamado em papo informal")

        monkeypatch.setattr(engine, "extrair_perfil", _extrair_perfil_nao_deveria_ser_chamado)
        monkeypatch.setattr(engine, "quer_nova_recomendacao", lambda texto: False)
        monkeypatch.setattr(engine, "precisa_busca", lambda texto: False)
        monkeypatch.setattr(engine, "_gerar_resposta_conversa", lambda texto: "Oi! Tudo bem?")

        def _answer_nao_deveria_ser_chamado(texto):
            raise AssertionError("answer() não deveria ser chamado em papo informal")

        monkeypatch.setattr(engine, "answer", _answer_nao_deveria_ser_chamado)

        sessao = _sessao(dict(_PERFIL_COMPLETO), fase="completo")
        resposta = responder("user-1", "oi, tudo bem?", sessao)

        assert resposta == "Oi! Tudo bem?"

    def test_falha_ao_gerar_resposta_de_conversa_cai_no_fallback(self, monkeypatch):
        monkeypatch.setattr(engine, "quer_nova_recomendacao", lambda texto: False)
        monkeypatch.setattr(engine, "precisa_busca", lambda texto: False)

        def _gerar_resposta_conversa_com_erro(texto):
            raise RuntimeError("Anthropic indisponível")

        monkeypatch.setattr(engine, "_gerar_resposta_conversa", _gerar_resposta_conversa_com_erro)

        sessao = _sessao(dict(_PERFIL_COMPLETO), fase="completo")
        resposta = responder("user-1", "oi", sessao)

        assert resposta == _MENSAGEM_FALLBACK

    def test_pedido_explicito_gera_nova_recomendacao_sem_chamar_rag(self, monkeypatch):
        """
        Pedido de nova recomendação passa a mensagem de novo pelo
        extrator antes de recomendar -- interesse/modalidade funcionam
        como sugestão atualizável, não um valor fixo desde a coleta.
        """
        monkeypatch.setattr(
            engine, "extrair_perfil", lambda texto, perfil_atual, historico=None: Perfil(**perfil_atual)
        )

        def _answer_nao_deveria_ser_chamado(texto):
            raise AssertionError("answer() não deveria ser chamado quando a pessoa pede nova recomendação")

        monkeypatch.setattr(engine, "quer_nova_recomendacao", lambda texto: True)
        monkeypatch.setattr(engine, "gerar_recomendacao", lambda perfil: "Que tal essa outra opção?")
        monkeypatch.setattr(engine, "answer", _answer_nao_deveria_ser_chamado)

        sessao = _sessao(dict(_PERFIL_COMPLETO), fase="completo")
        resposta = responder("user-1", "mostra outra opção", sessao)

        assert resposta == "Que tal essa outra opção?"

    def test_pedido_de_nova_recomendacao_atualiza_interesse_mencionado(self, monkeypatch):
        """
        Se a pessoa pede outra opção mencionando uma área diferente da
        que já estava salva, essa área nova chega em gerar_recomendacao
        e fica persistida na sessão -- não trava no interesse original.
        """
        capturado = {}

        def fake_extrair_perfil(texto, perfil_atual, historico=None):
            return Perfil(**{**perfil_atual, "interesse": "tecnologia"})

        def fake_gerar_recomendacao(perfil):
            capturado["interesse"] = perfil.interesse
            return "Achei opções de tecnologia pra você!"

        monkeypatch.setattr(engine, "extrair_perfil", fake_extrair_perfil)
        monkeypatch.setattr(engine, "quer_nova_recomendacao", lambda texto: True)
        monkeypatch.setattr(engine, "gerar_recomendacao", fake_gerar_recomendacao)

        sessao = _sessao(dict(_PERFIL_COMPLETO), fase="completo")
        resposta = responder("user-1", "tem opção de tecnologia?", sessao)

        assert resposta == "Achei opções de tecnologia pra você!"
        assert capturado["interesse"] == "tecnologia"
        assert sessao["perfil"]["interesse"] == "tecnologia"

    def test_falha_ao_gerar_nova_recomendacao_cai_no_fallback(self, monkeypatch):
        monkeypatch.setattr(
            engine, "extrair_perfil", lambda texto, perfil_atual, historico=None: Perfil(**perfil_atual)
        )

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


class TestReinicio:
    """
    Cobertura da integracao de reinicio em `responder()` -- a logica
    pura (`limpar_para_outra_area`, `perfil_zerado`,
    `classificar_pedido_reinicio`, `eh_confirmacao_positiva`) ja e
    testada isoladamente em `tests/test_reset.py`; aqui o foco e a
    orquestracao de fase (`fase_dialogo`/`fase_dialogo_anterior`) que
    só existe em `channels/engine.py`.
    """

    def test_perfil_incompleto_nao_chama_classificador_de_reinicio(self, monkeypatch):
        # Durante a coleta (perfil ainda incompleto -- vazio ou com
        # alguns campos), o classificador de reinicio nao roda: ele so
        # ve o texto solto, sem saber que pergunta o bot acabou de
        # fazer, e e pouco confiavel distinguindo "respondendo a
        # pergunta atual" de "pedindo reinicio" -- confirmado ao vivo,
        # respostas normais de interesse/alcance foram classificadas
        # incorretamente como buscar_outra_area, travando a coleta em
        # loop. Por isso so roda com o perfil ja completo.
        def _classificador_nao_deveria_ser_chamado(texto):
            raise AssertionError("classificar_pedido_reinicio não deveria ser chamado com perfil incompleto")

        monkeypatch.setattr(engine, "classificar_pedido_reinicio", _classificador_nao_deveria_ser_chamado)
        monkeypatch.setattr(engine, "extrair_perfil", lambda texto, perfil_atual, historico=None: Perfil(**perfil_atual))
        monkeypatch.setattr(engine, "_gerar_pergunta_coleta", lambda perfil: "Pergunta de coleta.")

        perfil_vazio = {c: None for c in ("cidade", "escolaridade", "interesse", "nivel", "modalidade", "alcance")}
        responder("user-1", "oi", _sessao(perfil_vazio))  # não deve levantar AssertionError
        responder("user-1", "moro em Blumenau", _sessao(dict(_PERFIL_INCOMPLETO)))  # idem

    def test_perfil_completo_chama_classificador_de_reinicio(self, monkeypatch):
        capturado = {}

        def fake_classificar(texto):
            capturado["chamado"] = True
            return "nenhum"

        monkeypatch.setattr(engine, "classificar_pedido_reinicio", fake_classificar)
        monkeypatch.setattr(engine, "quer_nova_recomendacao", lambda texto: False)
        monkeypatch.setattr(engine, "precisa_busca", lambda texto: False)
        monkeypatch.setattr(engine, "_gerar_resposta_conversa", lambda texto: "oi")

        sessao = _sessao(dict(_PERFIL_COMPLETO), fase="completo")
        responder("user-1", "quero começar de novo", sessao)

        assert capturado.get("chamado") is True

    def test_buscar_outra_area_preserva_cidade_e_pede_o_que_falta(self, monkeypatch):
        monkeypatch.setattr(engine, "classificar_pedido_reinicio", lambda texto: "buscar_outra_area")

        capturado = {}

        def fake_gerar_pergunta(perfil):
            capturado["perfil"] = perfil
            return "Legal, e que área te interessa agora?"

        monkeypatch.setattr(engine, "_gerar_pergunta_coleta", fake_gerar_pergunta)

        sessao = _sessao(dict(_PERFIL_COMPLETO), fase="completo")
        resposta = responder("user-1", "quero ver outra área", sessao)

        assert resposta == "Legal, e que área te interessa agora?"
        assert sessao["perfil"]["cidade"] == "Blumenau"
        assert sessao["perfil"]["interesse"] is None
        assert sessao["perfil"]["nivel"] is None
        assert sessao["fase_dialogo"] == "coletando"
        assert capturado["perfil"].cidade == "Blumenau"
        assert capturado["perfil"].interesse is None

    def test_falha_ao_gerar_pergunta_apos_buscar_outra_area_cai_no_fallback(self, monkeypatch):
        monkeypatch.setattr(engine, "classificar_pedido_reinicio", lambda texto: "buscar_outra_area")

        def _gerar_pergunta_com_erro(perfil):
            raise RuntimeError("Anthropic indisponível")

        monkeypatch.setattr(engine, "_gerar_pergunta_coleta", _gerar_pergunta_com_erro)

        sessao = _sessao(dict(_PERFIL_COMPLETO), fase="completo")
        resposta = responder("user-1", "quero ver outra área", sessao)

        assert resposta == engine._MENSAGEM_FALLBACK

    def test_comecar_de_novo_entra_em_confirmacao_e_guarda_fase_anterior(self, monkeypatch):
        monkeypatch.setattr(engine, "classificar_pedido_reinicio", lambda texto: "comecar_de_novo")
        monkeypatch.setattr(engine, "_gerar_confirmacao_reinicio", lambda texto: "Quer mesmo apagar tudo?")

        sessao = _sessao(dict(_PERFIL_COMPLETO), fase="completo")
        resposta = responder("user-1", "quero começar de novo", sessao)

        assert resposta == "Quer mesmo apagar tudo?"
        assert sessao["fase_dialogo"] == "confirmando_reinicio"
        assert sessao["fase_dialogo_anterior"] == "completo"
        # Perfil não é tocado até a confirmação de fato chegar.
        assert sessao["perfil"] == _PERFIL_COMPLETO

    def test_falha_ao_gerar_confirmacao_de_reinicio_cai_no_fallback_proprio(self, monkeypatch):
        monkeypatch.setattr(engine, "classificar_pedido_reinicio", lambda texto: "comecar_de_novo")

        def _gerar_confirmacao_com_erro(texto):
            raise RuntimeError("Anthropic indisponível")

        monkeypatch.setattr(engine, "_gerar_confirmacao_reinicio", _gerar_confirmacao_com_erro)

        sessao = _sessao(dict(_PERFIL_COMPLETO), fase="completo")
        resposta = responder("user-1", "quero começar de novo", sessao)

        assert resposta == engine._MENSAGEM_FALLBACK_CONFIRMACAO_REINICIO
        # Mesmo com falha na geração da pergunta, a sessão já entrou
        # em modo de confirmação -- senão a pessoa ficaria presa sem
        # nunca conseguir confirmar de fato.
        assert sessao["fase_dialogo"] == "confirmando_reinicio"

    def test_confirmacao_positiva_apaga_perfil_e_historico(self, monkeypatch):
        sessao = _sessao(dict(_PERFIL_COMPLETO), fase="confirmando_reinicio")
        sessao["fase_dialogo_anterior"] = "completo"
        sessao["historico"] = [{"de": "usuario", "texto": "oi"}, {"de": "bot", "texto": "olá!"}]

        resposta = responder("user-1", "sim", sessao)

        assert resposta == "Prontinho, apaguei tudo! Vamos começar de novo: em qual cidade você mora?"
        assert all(v is None for v in sessao["perfil"].values())
        assert sessao["fase_dialogo"] == "coletando"
        assert sessao["historico"] == []
        assert "fase_dialogo_anterior" not in sessao

    def test_confirmacao_negativa_restaura_fase_anterior_guardada(self, monkeypatch):
        sessao = _sessao(dict(_PERFIL_COMPLETO), fase="confirmando_reinicio")
        sessao["fase_dialogo_anterior"] = "coletando"

        resposta = responder("user-1", "não, deixa como está", sessao)

        assert resposta == "Sem problema, mantive seus dados como estavam."
        assert sessao["fase_dialogo"] == "coletando"
        assert sessao["perfil"] == _PERFIL_COMPLETO
        assert "fase_dialogo_anterior" not in sessao

    def test_confirmacao_negativa_sem_fase_anterior_guardada_cai_em_completo(self, monkeypatch):
        # Sessão antiga no Redis, de antes do campo fase_dialogo_anterior
        # existir -- fallback defensivo pra "completo".
        sessao = _sessao(dict(_PERFIL_COMPLETO), fase="confirmando_reinicio")

        resposta = responder("user-1", "não", sessao)

        assert resposta == "Sem problema, mantive seus dados como estavam."
        assert sessao["fase_dialogo"] == "completo"


_PERFIL_SO_FALTA_NIVEL = {
    "cidade": "Blumenau",
    "escolaridade": "ensino medio completo",
    "interesse": "tecnologia",
    "nivel": None,
    "modalidade": None,
    "alcance": "regional",
}


class TestBotoesDeNivel:
    def test_pergunta_de_nivel_vem_com_botoes_quando_e_o_proximo_campo(self, monkeypatch):
        monkeypatch.setattr(engine, "_gerar_pergunta_coleta", lambda perfil: "Qual nível você quer?")

        sessao = _sessao(dict(_PERFIL_SO_FALTA_NIVEL))
        resposta = responder("user-1", "quero tecnologia", sessao)

        assert resposta == "Qual nível você quer?"
        assert resposta.botoes == engine._botoes_nivel(_PERFIL_SO_FALTA_NIVEL["escolaridade"])

    def test_pergunta_de_outro_campo_nao_vem_com_botoes(self, monkeypatch):
        monkeypatch.setattr(engine, "_gerar_pergunta_coleta", lambda perfil: "Qual sua escolaridade?")

        perfil_faltando_escolaridade = {**_PERFIL_SO_FALTA_NIVEL, "escolaridade": None, "alcance": None}
        sessao = _sessao(perfil_faltando_escolaridade)
        resposta = responder("user-1", "moro em Blumenau", sessao)

        assert resposta == "Qual sua escolaridade?"
        assert getattr(resposta, "botoes", None) is None

    def test_nivel_escolhido_ignora_extrator_e_define_nivel_direto(self, monkeypatch):
        def _extrair_perfil_nao_deveria_ser_chamado(texto, perfil_atual, historico=None):
            raise AssertionError("extrair_perfil() não deveria ser chamado com nivel_escolhido")

        monkeypatch.setattr(engine, "extrair_perfil", _extrair_perfil_nao_deveria_ser_chamado)
        monkeypatch.setattr(engine, "gerar_recomendacao", lambda perfil: "Achei um curso pra você!")

        sessao = _sessao(dict(_PERFIL_SO_FALTA_NIVEL))
        resposta = responder("user-1", "2", sessao, nivel_escolhido="tecnico subsequente")

        assert resposta == "Achei um curso pra você!"
        assert sessao["perfil"]["nivel"] == "tecnico subsequente"
        assert sessao["fase_dialogo"] == "completo"

    def test_nivel_escolhido_pula_classificador_de_reinicio(self, monkeypatch):
        def _classificador_nao_deveria_ser_chamado(texto):
            raise AssertionError("classificar_pedido_reinicio não deveria ser chamado com nivel_escolhido")

        monkeypatch.setattr(engine, "classificar_pedido_reinicio", _classificador_nao_deveria_ser_chamado)
        monkeypatch.setattr(engine, "gerar_recomendacao", lambda perfil: "Achei um curso pra você!")

        sessao = _sessao(dict(_PERFIL_SO_FALTA_NIVEL))
        responder("user-1", "2", sessao, nivel_escolhido="superior")  # não deve levantar AssertionError


class TestBotoesDeReinicio:
    def test_confirmacao_de_reinicio_vem_com_botoes(self, monkeypatch):
        monkeypatch.setattr(engine, "classificar_pedido_reinicio", lambda texto: "comecar_de_novo")
        monkeypatch.setattr(engine, "_gerar_confirmacao_reinicio", lambda texto: "Quer mesmo apagar tudo?")

        sessao = _sessao(dict(_PERFIL_COMPLETO), fase="completo")
        resposta = responder("user-1", "quero começar de novo", sessao)

        assert resposta == "Quer mesmo apagar tudo?"
        assert resposta.botoes == engine._BOTOES_REINICIO

    def test_fallback_de_confirmacao_de_reinicio_tambem_vem_com_botoes(self, monkeypatch):
        monkeypatch.setattr(engine, "classificar_pedido_reinicio", lambda texto: "comecar_de_novo")

        def _gerar_confirmacao_com_erro(texto):
            raise RuntimeError("Anthropic indisponível")

        monkeypatch.setattr(engine, "_gerar_confirmacao_reinicio", _gerar_confirmacao_com_erro)

        sessao = _sessao(dict(_PERFIL_COMPLETO), fase="completo")
        resposta = responder("user-1", "quero começar de novo", sessao)

        assert resposta == engine._MENSAGEM_FALLBACK_CONFIRMACAO_REINICIO
        assert resposta.botoes == engine._BOTOES_REINICIO


def _sessao_inicio() -> dict:
    return {"perfil": {}, "fase_dialogo": "inicio", "historico": []}


class TestBifurcacaoInicial:
    """
    Sessao nova (fase_dialogo == "inicio") nunca resolveu a
    bifurcacao entre buscar curso, tirar duvida, ou nao decidiu ainda
    -- ver `channels/engine.py::responder`.
    """

    def test_pergunta_direta_pula_o_menu_e_responde_via_rag(self, monkeypatch):
        def _extrair_perfil_nao_deveria_ser_chamado(texto, perfil_atual, historico=None):
            raise AssertionError("extrair_perfil não deveria ser chamado numa pergunta direta")

        monkeypatch.setattr(engine, "extrair_perfil", _extrair_perfil_nao_deveria_ser_chamado)
        monkeypatch.setattr(engine, "quer_nova_recomendacao", lambda texto: False)
        monkeypatch.setattr(engine, "precisa_busca", lambda texto: True)
        monkeypatch.setattr(engine, "answer", lambda texto: {"answer": "A inscrição fecha em 20/08.", "sources": []})

        sessao = _sessao_inicio()
        resposta = responder("user-1", "quando fecha a inscrição do edital X?", sessao)

        assert "A inscrição fecha em 20/08." in resposta
        assert sessao["fase_dialogo"] == "conversa_livre"

    def test_texto_ambiguo_mostra_o_menu_inicial(self, monkeypatch):
        monkeypatch.setattr(engine, "quer_nova_recomendacao", lambda texto: False)
        monkeypatch.setattr(engine, "precisa_busca", lambda texto: False)

        sessao = _sessao_inicio()
        resposta = responder("user-1", "oi", sessao)

        assert resposta == engine._MENSAGEM_MENU_INICIAL
        assert resposta.botoes == engine._BOTOES_INICIO
        assert sessao["fase_dialogo"] == "inicio"

    def test_texto_livre_de_pedido_de_curso_inicia_coleta(self, monkeypatch):
        monkeypatch.setattr(engine, "quer_nova_recomendacao", lambda texto: True)
        monkeypatch.setattr(engine, "extrair_perfil", lambda texto, perfil_atual, historico=None: Perfil(**perfil_atual))
        monkeypatch.setattr(engine, "_gerar_pergunta_coleta", lambda perfil: "Em qual cidade você mora?")

        sessao = _sessao_inicio()
        resposta = responder("user-1", "quero achar um curso", sessao)

        assert resposta == "Em qual cidade você mora?"
        assert sessao["fase_dialogo"] == "coletando"

    def test_botao_buscar_curso_inicia_coleta_sem_chamar_classificador(self, monkeypatch):
        def _classificador_nao_deveria_ser_chamado(texto):
            raise AssertionError("quer_nova_recomendacao não deveria ser chamado pro botão sintético")

        monkeypatch.setattr(engine, "quer_nova_recomendacao", _classificador_nao_deveria_ser_chamado)
        monkeypatch.setattr(engine, "extrair_perfil", lambda texto, perfil_atual, historico=None: Perfil(**perfil_atual))
        monkeypatch.setattr(engine, "_gerar_pergunta_coleta", lambda perfil: "Em qual cidade você mora?")

        sessao = _sessao_inicio()
        resposta = responder("user-1", "quero buscar um curso", sessao)

        assert resposta == "Em qual cidade você mora?"
        assert sessao["fase_dialogo"] == "coletando"

    def test_botao_tenho_duvida_entra_em_conversa_livre_sem_chamar_classificador(self, monkeypatch):
        def _classificador_nao_deveria_ser_chamado(texto):
            raise AssertionError("nenhum classificador deveria ser chamado pro botão sintético de dúvida")

        monkeypatch.setattr(engine, "quer_nova_recomendacao", _classificador_nao_deveria_ser_chamado)
        monkeypatch.setattr(engine, "precisa_busca", _classificador_nao_deveria_ser_chamado)

        sessao = _sessao_inicio()
        resposta = responder("user-1", "tenho uma duvida", sessao)

        assert resposta == engine._MENSAGEM_CONVITE_DUVIDA
        assert sessao["fase_dialogo"] == "conversa_livre"

    def test_conversa_livre_ja_estabelecida_responde_via_rag_sem_extrair_perfil(self, monkeypatch):
        def _extrair_perfil_nao_deveria_ser_chamado(texto, perfil_atual, historico=None):
            raise AssertionError("extrair_perfil não deveria ser chamado em conversa_livre")

        monkeypatch.setattr(engine, "extrair_perfil", _extrair_perfil_nao_deveria_ser_chamado)
        monkeypatch.setattr(engine, "quer_nova_recomendacao", lambda texto: False)
        monkeypatch.setattr(engine, "precisa_busca", lambda texto: True)
        monkeypatch.setattr(engine, "answer", lambda texto: {"answer": "resposta do RAG", "sources": []})

        sessao = {"perfil": {}, "fase_dialogo": "conversa_livre", "historico": []}
        resposta = responder("user-1", "quais documentos preciso?", sessao)

        assert "resposta do RAG" in resposta
        assert sessao["fase_dialogo"] == "conversa_livre"

    def test_saida_de_escape_migra_conversa_livre_pra_coleta(self, monkeypatch):
        def _gerar_recomendacao_nao_deveria_ser_chamado(perfil):
            raise AssertionError("gerar_recomendacao não deveria ser chamado com perfil incompleto")

        monkeypatch.setattr(engine, "gerar_recomendacao", _gerar_recomendacao_nao_deveria_ser_chamado)
        monkeypatch.setattr(engine, "quer_nova_recomendacao", lambda texto: True)
        monkeypatch.setattr(
            engine,
            "extrair_perfil",
            lambda texto, perfil_atual, historico=None: Perfil(cidade="Blumenau"),
        )
        monkeypatch.setattr(engine, "_gerar_pergunta_coleta", lambda perfil: "Qual sua escolaridade?")

        sessao = {"perfil": {}, "fase_dialogo": "conversa_livre", "historico": []}
        resposta = responder("user-1", "quero uma recomendação de curso", sessao)

        assert resposta == "Qual sua escolaridade?"
        assert sessao["fase_dialogo"] == "coletando"
        assert sessao["perfil"]["cidade"] == "Blumenau"


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
