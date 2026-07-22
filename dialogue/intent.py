"""
Roteamento de intenção antes do RAG: decide se uma mensagem precisa de
busca nos editais (`retrieval.generate.answer`) ou é papo informal que
não precisa de retrieval nem geração ancorada nenhuma.

Duas camadas, da mais barata pra mais cara:

1. Regex com palavras-chave do domínio de editais -- bateu, é BUSCA,
   sem chamar LLM nenhum.
2. Se não bateu keyword, um classificador leve (Anthropic, mesmo
   padrão de `dialogue.recommendation.quer_nova_recomendacao`) decide
   entre BUSCA e CONVERSA.

Existe pra evitar o custo (Weaviate + geração ancorada) de rodar o
pipeline de RAG inteiro só pra uma saudação ou agradecimento -- e
complementa, não substitui, a auto-avaliação `recusa` que o próprio
RAG já faz em `retrieval/generate.py` pro que passar despercebido por
aqui.
"""

import logging
import re
import unicodedata

import anthropic
from pydantic import BaseModel

from config.prompts import PROMPT_CLASSIFICA_INTENCAO_BUSCA
from config.settings import settings

logger = logging.getLogger(__name__)

# Palavras-chave do domínio de editais -- constante solta no topo do
# módulo de propósito, pra ser fácil de editar sem mexer na lógica.
# Comparação é sem acento e sem caixa (ver `_normaliza`), então escreva
# as entradas aqui sempre sem acento.
_PALAVRAS_CHAVE_BUSCA: tuple[str, ...] = (
    "edital",
    "prazo",
    "inscricao",
    "documento",
    "requisito",
    "bolsa",
    "vaga",
    "cronograma",
    "resultado",
    "ifsc",
    "matricula",
    "selecao",
)

_CUMPRIMENTOS_INFORMAL: tuple[str, ...] = (
    "oi",
    "ola",
    "olá",
    "bom dia",
    "boa tarde",
    "boa noite",
    "e ai",
    "eae",
    "opa",
    "ei",
    "tudo bem",
)

_AGRADECIAMENTOS_INFORMAL: tuple[str, ...] = (
    "obrigado",
    "obrigada",
    "valeu",
    "brigado",
    "muito obrigado",
    "muito obrigada",
)

_REGEX_BUSCA = re.compile(
    r"\b(" + "|".join(re.escape(p) for p in _PALAVRAS_CHAVE_BUSCA) + r")\b"
)


class _ClassificacaoIntencao(BaseModel):
    precisa_busca: bool


def _normaliza(texto: str) -> str:
    """Normaliza texto para comparação: minúsculo e sem acento."""
    sem_acento = unicodedata.normalize("NFKD", texto).encode("ascii", "ignore").decode()
    return sem_acento.strip().lower()


def _bate_keyword(texto: str) -> bool:
    """Camada 1: regex sobre palavras-chave do domínio de editais.

    Parameters
    ----------
    texto : str
        Mensagem da pessoa.

    Returns
    -------
    bool
        True se alguma palavra-chave de `_PALAVRAS_CHAVE_BUSCA`
        aparecer na mensagem (sem acento, sem caixa, respeitando
        fronteira de palavra).
    """
    return bool(_REGEX_BUSCA.search(_normaliza(texto)))


def _eh_papo_informal(texto: str) -> bool:
    """Detecta saudações, agradecimentos e outras mensagens triviais.

    Essas mensagens não devem cair no pipeline de RAG nem citar fontes.
    """
    texto_norm = _normaliza(texto)
    if not texto_norm:
        return True
    if not re.search(r"[a-z0-9]", texto_norm):
        return True
    if texto_norm in _CUMPRIMENTOS_INFORMAL or texto_norm in _AGRADECIAMENTOS_INFORMAL:
        return True

    tokens = re.findall(r"[a-z0-9]+", texto_norm)
    allowed_tokens = {
        *[t for t in _CUMPRIMENTOS_INFORMAL if " " not in t],
        *[t for t in _AGRADECIAMENTOS_INFORMAL if " " not in t],
        "tudo",
        "bem",
        "muito",
        "bom",
        "dia",
        "boa",
        "tarde",
        "noite",
        "e",
        "ai",
        "opa",
        "ei",
    }
    return len(tokens) <= 4 and all(token in allowed_tokens for token in tokens)


def _chamar_llm_classificador(texto: str) -> dict:
    """Camada 2: classificador leve via Anthropic (Haiku).

    Isolado numa função própria pra poder ser trocado/mockado nos
    testes sem precisar de chave de API de verdade -- mesmo padrão de
    `dialogue.recommendation._chamar_llm_classificador`.

    Parameters
    ----------
    texto : str
        Mensagem da pessoa.

    Returns
    -------
    dict
        Dump do `_ClassificacaoIntencao` (`{"precisa_busca": bool}`).
    """
    client = anthropic.Anthropic()
    resposta = client.messages.parse(
        model=settings.anthropic_model_extracao,
        max_tokens=20,
        temperature=0,
        system=PROMPT_CLASSIFICA_INTENCAO_BUSCA,
        messages=[{"role": "user", "content": texto}],
        output_format=_ClassificacaoIntencao,
    )
    return resposta.parsed_output.model_dump()


def precisa_busca(texto: str) -> bool:
    """Decide se a mensagem precisa do pipeline de RAG (busca) ou não.

    Camada 1 (regex) decide sozinha quando bate; só cai pro
    classificador (camada 2) quando nenhuma palavra-chave aparece.
    Falha no classificador não bloqueia a conversa -- mas ao contrário
    de `quer_nova_recomendacao` (que assume False na dúvida), aqui o
    lado seguro é assumir que precisa buscar: é bem pior deixar de
    responder uma pergunta real sobre edital do que rodar uma busca à
    toa.

    Parameters
    ----------
    texto : str
        Mensagem da pessoa.

    Returns
    -------
    bool
        True se a mensagem precisa do pipeline de RAG.
    """
    if _bate_keyword(texto):
        return True

    if _eh_papo_informal(texto):
        return False

    try:
        resultado = _chamar_llm_classificador(texto)
    except Exception as exc:
        logger.error("Falha ao classificar intenção de busca (%s)", type(exc).__name__)
        return True

    return bool(resultado.get("precisa_busca", True))
