"""
Logica de reinicio do perfil, com duas granularidades:

- "buscar_outra_area": preserva cidade, escolaridade e alcance; limpa
  interesse, nivel e modalidade. Retoma a coleta a partir do que falta.
- "comecar_de_novo": limpa o perfil inteiro e o historico da sessao.
  Exige confirmacao antes de aplicar (ver `fase_dialogo ==
  "confirmando_reinicio"` no engine.py).

Ambas sao roteadas por texto livre, via `classificar_pedido_reinicio`.
A confirmacao de "comecar_de_novo" tambem pode vir de um botao inline
do Telegram (ver `channels/telegram.py` e `channels/engine.py`) -- o
botao usa os textos sinteticos `TEXTO_SINTETICO_CONFIRMAR`/
`TEXTO_SINTETICO_CANCELAR` definidos abaixo, que passam pela mesma
`eh_confirmacao_positiva()` de sempre, sem nenhuma logica de decisao
separada pro caminho de botao.
"""

import logging
import re

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


_CONFIRMACOES_POSITIVAS = frozenset(
    {
        "sim", "s", "confirmo", "confirmado", "isso", "isso mesmo", "pode",
        "pode sim", "sim pode", "sim, pode", "quero", "certeza", "com certeza",
        "positivo", "manda ver", "yes",
    }
)

# So remove pontuacao final repetida (ex.: "sim!", "sim...", "pode?") --
# a frase inteira ainda precisa bater com uma das confirmacoes exatas
# acima. Match parcial por palavra ficaria ambiguo demais pra decisao
# tao sensivel quanto essa: "quero" sozinho confirma, mas "quero pensar"
# claramente nao, e as duas compartilham a palavra "quero".
_PONTUACAO_FINAL = re.compile(r"[!.,;?]+$")


def eh_confirmacao_positiva(texto: str) -> bool:
    """
    Reconhece uma confirmacao simples de "sim" pro reinicio total.
    Deliberadamente simples (sem LLM) -- e uma decisao binaria de
    baixo risco, nao precisa de classificador caro. Tolera pontuacao
    final e algumas variacoes comuns de frase inteira (ex.: "Sim!",
    "isso mesmo", "sim, pode"), mas nunca casa uma palavra afirmativa
    isolada dentro de uma frase maior -- qualquer coisa fora da lista
    exata (inclusive "quero pensar") e tratada como recusa.
    """
    texto_normalizado = _PONTUACAO_FINAL.sub("", texto.strip().lower()).strip()
    return texto_normalizado in _CONFIRMACOES_POSITIVAS


# Contrato explicito pro botao inline de confirmacao de reinicio
# (`channels/telegram.py`/`channels/engine.py`): callback_data que o
# Telegram devolve, e o "texto sintetico" que cada um mapeia -- em vez
# do canal de Telegram ter que adivinhar uma string magica que bate
# com `_CONFIRMACOES_POSITIVAS`, o proprio modulo de reinicio exporta
# os valores garantidos. O teste de pares logo abaixo (ver
# tests/test_reset.py) fixa essa ponte: se `_CONFIRMACOES_POSITIVAS`
# mudar sem essa constante ser atualizada junto, o teste quebra em vez
# do botao falhar silenciosamente em producao.
CALLBACK_REINICIO_CONFIRMAR = "reinicio:confirmar"
CALLBACK_REINICIO_CANCELAR = "reinicio:cancelar"
TEXTO_SINTETICO_CONFIRMAR = "confirmo"
TEXTO_SINTETICO_CANCELAR = "manter meus dados"