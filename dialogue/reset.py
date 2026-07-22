"""
Logica de reinicio do perfil, com duas granularidades:

- "buscar_outra_area": preserva cidade, escolaridade e alcance; limpa
  interesse, nivel e modalidade. Retoma a coleta a partir do que falta.
- "comecar_de_novo": limpa o perfil inteiro e o historico da sessao.
  Exige confirmacao antes de aplicar (ver `fase_dialogo ==
  "confirmando_reinicio"` no engine.py).

Ambas sao roteadas por texto livre. "buscar_outra_area" e uma variante
mais ambigua de "comecar_de_novo" so passam por
`classificar_pedido_reinicio` (classificador probabilistico, LLM) --
restrito a perfil ja completo, por ser pouco confiavel durante a coleta
(ver comentario em `channels/engine.py::responder`). Ja "comecar_de_novo"
tem tambem uma camada rapida e deterministica (regex, sem LLM):
`eh_gatilho_explicito_de_reinicio_total`, disponivel em QUALQUER fase da
conversa -- coleta, RAG ou perfil completo -- com prioridade sobre o
roteamento normal.

A confirmacao de "comecar_de_novo" tambem pode vir de um botao inline
do Telegram (ver `channels/telegram.py` e `channels/engine.py`) -- o
botao usa os textos sinteticos `TEXTO_SINTETICO_CONFIRMAR`/
`TEXTO_SINTETICO_CANCELAR` definidos abaixo, que passam pela mesma
`eh_confirmacao_positiva()` de sempre, sem nenhuma logica de decisao
separada pro caminho de botao.
"""

import logging
import re
import unicodedata

import anthropic

from config.prompts import PROMPT_CLASSIFICA_REINICIO
from config.settings import settings

logger = logging.getLogger(__name__)


def _normaliza(texto: str) -> str:
    """Normaliza texto para comparação: minúsculo e sem acento."""
    sem_acento = unicodedata.normalize("NFKD", texto).encode("ascii", "ignore").decode()
    return sem_acento.strip().lower()


# Gatilho explícito de reinício total -- palavra-chave principal e
# variações fortes o suficiente pra disparar sem depender do
# classificador (regex, sem LLM, mesmo padrão em duas camadas de
# `dialogue/intent.py`). "recomeçar" é a palavra-gatilho principal
# (evita "menu", ambíguo com pedido de ver as opções de um teclado).
# Radicais (\w*), não frase fixa -- cobre conjugação comum em portugues
# ("recomecar", "recomeca", "recomecando"; "comecar de novo", "comeca
# de novo") sem precisar listar cada forma. Só entram aqui radicais que
# só fazem sentido como reinício -- nada de palavra solta que possa
# aparecer numa resposta normal de coleta.
_REGEX_GATILHO_REINICIO_TOTAL = re.compile(
    r"\brecomec\w*\b"
    r"|\bcomec\w*\s+de\s+novo\b"
    r"|\breinici\w*\b"
    r"|\besquece\s+tudo\b"
)


def eh_gatilho_explicito_de_reinicio_total(texto: str) -> bool:
    """
    Camada rápida (sem LLM) de reconhecimento de "comecar_de_novo",
    disponível a qualquer momento da conversa -- durante a coleta de
    perfil, durante uma pergunta ao RAG, ou depois de uma recomendação.
    Tem prioridade sobre o roteamento normal: detectado, interrompe o
    que estiver acontecendo (mesmo uma pergunta específica de coleta) e
    parte pra confirmação, sem precisar que o perfil já esteja completo
    -- ao contrário de `classificar_pedido_reinicio` (probabilístico,
    mantido restrito a perfil completo pelo motivo documentado em
    `channels/engine.py::responder`).
    """
    return bool(_REGEX_GATILHO_REINICIO_TOTAL.search(_normaliza(texto)))

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