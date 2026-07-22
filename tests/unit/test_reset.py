"""
Testes puros do modulo de reinicio -- nao tocam Anthropic de verdade.
"""

import pytest

from dialogue import reset
from dialogue.reset import (
    TEXTO_SINTETICO_CANCELAR,
    TEXTO_SINTETICO_CONFIRMAR,
    classificar_pedido_reinicio,
    eh_confirmacao_positiva,
    eh_gatilho_explicito_de_reinicio_total,
    limpar_para_outra_area,
    perfil_zerado,
)


def test_buscar_outra_area_preserva_campos_certos():
    perfil_atual = {
        "cidade": "Joinville",
        "escolaridade": "ensino medio completo",
        "interesse": "mecanica",
        "nivel": "tecnico integrado",
        "modalidade": "presencial",
        "alcance": "regional",
    }
    resultado = limpar_para_outra_area(perfil_atual)

    assert resultado["cidade"] == "Joinville"
    assert resultado["escolaridade"] == "ensino medio completo"
    assert resultado["alcance"] == "regional"
    assert resultado["interesse"] is None
    assert resultado["nivel"] is None
    assert resultado["modalidade"] is None


def test_perfil_zerado_limpa_tudo():
    resultado = perfil_zerado()
    assert all(v is None for v in resultado.values())


def test_classificador_reconhece_buscar_outra_area(monkeypatch):
    monkeypatch.setattr(reset, "_chamar_classificador", lambda texto: "buscar_outra_area")
    assert classificar_pedido_reinicio("quero ver outra area") == "buscar_outra_area"


def test_classificador_reconhece_comecar_de_novo(monkeypatch):
    monkeypatch.setattr(reset, "_chamar_classificador", lambda texto: "comecar_de_novo")
    assert classificar_pedido_reinicio("esquece tudo, vamos recomecar") == "comecar_de_novo"


def test_classificador_reconhece_nenhum(monkeypatch):
    monkeypatch.setattr(reset, "_chamar_classificador", lambda texto: "nenhum")
    assert classificar_pedido_reinicio("quando fecha a inscricao?") == "nenhum"


def test_classificador_falha_devolve_nenhum(monkeypatch):
    def fake_com_erro(texto):
        raise RuntimeError("Anthropic indisponivel")

    monkeypatch.setattr(reset, "_chamar_classificador", fake_com_erro)
    assert classificar_pedido_reinicio("qualquer coisa") == "nenhum"


def test_classificador_valor_inesperado_devolve_nenhum(monkeypatch):
    monkeypatch.setattr(reset, "_chamar_classificador", lambda texto: "algo_estranho")
    assert classificar_pedido_reinicio("oi") == "nenhum"


def test_confirmacao_positiva_reconhece_variacoes():
    assert eh_confirmacao_positiva("sim") is True
    assert eh_confirmacao_positiva("Sim") is True
    assert eh_confirmacao_positiva(" confirmo ") is True
    assert eh_confirmacao_positiva("nao") is False
    assert eh_confirmacao_positiva("quero pensar") is False


def test_confirmacao_positiva_tolera_pontuacao_final():
    assert eh_confirmacao_positiva("Sim!") is True
    assert eh_confirmacao_positiva("pode?") is True
    assert eh_confirmacao_positiva("confirmado.") is True
    assert eh_confirmacao_positiva("sim...") is True


def test_confirmacao_positiva_reconhece_frases_naturais():
    assert eh_confirmacao_positiva("sim, pode") is True
    assert eh_confirmacao_positiva("pode sim") is True
    assert eh_confirmacao_positiva("isso mesmo") is True
    assert eh_confirmacao_positiva("com certeza") is True


def test_confirmacao_positiva_nao_casa_palavra_isolada_em_frase_maior():
    # "quero" sozinho confirma, mas embutido numa frase com outra
    # intencao (recusa/duvida) nao pode contar como confirmacao --
    # word-level match seria ambiguo demais aqui.
    assert eh_confirmacao_positiva("quero pensar mais um pouco") is False
    assert eh_confirmacao_positiva("acho que sim, mas nao tenho certeza") is False


def test_texto_sintetico_confirmar_e_reconhecido_como_positivo():
    # Pin: se _CONFIRMACOES_POSITIVAS mudar e parar de aceitar isso, o
    # botao de reinicio quebra silenciosamente em producao.
    assert eh_confirmacao_positiva(TEXTO_SINTETICO_CONFIRMAR) is True


def test_texto_sintetico_cancelar_e_reconhecido_como_negativo():
    assert eh_confirmacao_positiva(TEXTO_SINTETICO_CANCELAR) is False


@pytest.mark.parametrize(
    "texto",
    [
        "recomeçar",
        "/recomecar",
        "Recomeçar",
        "quero recomeçar",
        "quero começar de novo",
        "apaga meus dados e comeca de novo",  # conjugação "comeca", não "comecar"
        "esquece tudo",
        "reinicia",
        "reiniciar",
        "vamos reiniciar do zero",
        "reinicia tudo",
    ],
)
def test_gatilho_de_reinicio_total_reconhece_variacoes(texto):
    assert eh_gatilho_explicito_de_reinicio_total(texto) is True


@pytest.mark.parametrize(
    "texto",
    [
        "quero tecnologia",
        "Florianopolis",
        "topo ir pra uma cidade proxima",
        "ensino medio",
        "sim",
        "quando fecha a inscricao?",
        "",
    ],
)
def test_gatilho_de_reinicio_total_nao_falso_positivo_em_resposta_normal_de_coleta(texto):
    # Mesmas respostas que já causaram falso positivo no classificador
    # probabilístico (ver comentário em channels/engine.py::responder)
    # -- o gatilho rápido não pode repetir esse erro.
    assert eh_gatilho_explicito_de_reinicio_total(texto) is False