"""
Testes puros do modulo de reinicio -- nao tocam Anthropic de verdade.
"""

from dialogue import reset
from dialogue.reset import (
    classificar_pedido_reinicio,
    eh_confirmacao_positiva,
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