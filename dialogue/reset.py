"""
Logica de reinicio do perfil, com duas granularidades:

- "buscar_outra_area": preserva cidade, escolaridade e alcance; limpa
  interesse, nivel e modalidade. Retoma a coleta a partir do que falta.
- "comecar_de_novo": limpa o perfil inteiro e o historico da sessao.
  Exige confirmacao antes de aplicar (ver `fase_dialogo ==
  "confirmando_reinicio"` no engine.py).

Ambas sao roteadas tanto por botao quanto por texto livre, via
`classificar_pedido_reinicio`.
"""

import logging

import anthropic

from config.settings import settings
from dialogue.prompts import PROMPT_CLASSIFICA_REINICIO

logger = logging.getLogger(__name__)

_CAMPOS_PRESERVADOS_OUTRA_AREA = ("cidade", "escolaridade", "alcance")
_TODOS_OS_CAMPOS_PERFIL = (
    "cidade",
    "escolaridade",
    "interesse",
    "nivel",
    "modalidade",
    "alcance",
)


def limpar_para_outra_area(perfil_atual: dict) -> dict:
    """
    Preserva cidade, escolaridade e alcance; zera interesse, nivel e
    modalidade -- pra retomar a coleta a partir do que ainda falta,
    sem repetir o que a pessoa ja informou.
    """
    novo = {campo: None for campo in _TODOS_OS_CAMPOS_PERFIL}
    for campo in _CAMPOS_PRESERVADOS_OUTRA_AREA:
        novo[campo] = perfil_atual.get(campo)
    return novo


def perfil_zerado() -> dict:
    """Perfil inteiramente vazio, pro reinicio total."""
    return {campo: None for campo in _TODOS_OS_CAMPOS_PERFIL}


def _chamar_classificador(texto: str) -> str:
    """
    Isolado numa funcao propria pra poder ser mockado nos testes sem
    precisar de chave de API de verdade.
    """
    client = anthropic.Anthropic()
    resposta = client.messages.create(
        model=settings.anthropic_model_extracao,
        max_tokens=20,
        system=PROMPT_CLASSIFICA_REINICIO,
        messages=[{"role": "user", "content": texto}],
    )
    return next(b.text for b in resposta.content if b.type == "text").strip()


def classificar_pedido_reinicio(texto: str) -> str:
    """
    Devolve "buscar_outra_area", "comecar_de_novo" ou "nenhum".

    Em caso de falha na chamada ao LLM, devolve "nenhum" -- falha
    aberta consciente: melhor deixar a mensagem seguir pro fluxo
    normal do que reiniciar um perfil por engano.
    """
    try:
        resultado = _chamar_classificador(texto)
    except Exception as exc:
        logger.error("Falha ao classificar pedido de reinicio (%s)", type(exc).__name__)
        return "nenhum"

    if resultado not in ("buscar_outra_area", "comecar_de_novo", "nenhum"):
        logger.error("Classificador de reinicio devolveu valor inesperado: %r", resultado)
        return "nenhum"
    return resultado


def eh_confirmacao_positiva(texto: str) -> bool:
    """
    Reconhece uma confirmacao simples de "sim" pro reinicio total.
    Deliberadamente simples (sem LLM) -- e uma decisao binaria de
    baixo risco, nao precisa de classificador caro.
    """
    afirmativos = {"sim", "s", "confirmo", "isso", "pode", "quero", "yes"}
    return texto.strip().lower() in afirmativos