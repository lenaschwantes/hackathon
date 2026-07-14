"""
Ponte entre o perfil coletado e o motor de recomendacao estruturado
(`recommend/opportunities.py`), com a redacao final feita pelo LLM.

O corte de cidade/modalidade/calendario e sempre feito por
`recomendar()` -- puro, determinístico, sem LLM. O LLM só recebe o
resultado já pronto e redige a mensagem; nunca decide data nem
inventa curso, igual ao contrato documentado em `recomendar()`.
"""

import json
import logging
import os
from datetime import date

from openai import OpenAI

from dialogue.profile import Perfil
from dialogue.prompts import PROMPT_CLASSIFICA_PEDIDO_RECOMENDACAO, PROMPT_RECOMENDACAO
from recommend.opportunities import recomendar

logger = logging.getLogger(__name__)


def montar_contexto(perfil: Perfil, hoje: date) -> dict:
    """
    Chama o motor de recomendacao e serializa o resultado num dict
    JSON-safe (datas viram string ISO) pronto pra entrar no prompt.
    """
    resultado = recomendar(
        cidade=perfil.cidade, hoje=hoje, nivel=perfil.nivel, modalidade=perfil.modalidade
    )
    return {
        "interesse": perfil.interesse,
        "abertas": [o.model_dump(mode="json") for o in resultado["abertas"]],
        "proxima": resultado["proxima"].model_dump(mode="json") if resultado["proxima"] else None,
    }


def _chamar_llm(contexto: dict) -> str:
    """
    Isolado numa funcao propria pra poder ser trocado/mockado nos
    testes sem precisar de chave de API de verdade.
    """
    client = OpenAI(
        api_key=os.environ["GROQ_API_KEY"],
        base_url="https://api.groq.com/openai/v1",
    )
    prompt = PROMPT_RECOMENDACAO.format(contexto=json.dumps(contexto, ensure_ascii=False))
    resposta = client.chat.completions.create(
        model=os.environ.get("GROQ_MODEL", "llama-3.3-70b-versatile"),
        messages=[{"role": "system", "content": prompt}],
        temperature=0.7,
    )
    return resposta.choices[0].message.content


def gerar_recomendacao(perfil: Perfil, hoje: date | None = None) -> str:
    """
    Monta o contexto a partir do perfil completo e pede ao LLM que
    redija a recomendacao final pro cidadao.
    """
    contexto = montar_contexto(perfil, hoje or date.today())
    return _chamar_llm(contexto)


def _chamar_llm_classificador(texto: str) -> dict:
    """
    Isolado numa funcao propria pra poder ser trocado/mockado nos
    testes sem precisar de chave de API de verdade -- mesmo motivo de
    `_chamar_llm` acima.
    """
    client = OpenAI(
        api_key=os.environ["GROQ_API_KEY"],
        base_url="https://api.groq.com/openai/v1",
    )
    resposta = client.chat.completions.create(
        model=os.environ.get("GROQ_MODEL", "llama-3.3-70b-versatile"),
        messages=[
            {"role": "system", "content": PROMPT_CLASSIFICA_PEDIDO_RECOMENDACAO},
            {"role": "user", "content": texto},
        ],
        response_format={"type": "json_object"},
        temperature=0,
    )
    return json.loads(resposta.choices[0].message.content)


def quer_nova_recomendacao(texto: str) -> bool:
    """
    Com o perfil ja completo, decide se a mensagem e um pedido
    explicito por outra recomendacao (ex: "mostra outra opcao") em vez
    de uma pergunta normal sobre o que ja foi recomendado. Falha do
    classificador nao bloqueia a conversa -- na duvida, segue pro RAG
    normal (mesma filosofia de fallback de `extrair_perfil`).
    """
    try:
        resultado = _chamar_llm_classificador(texto)
    except Exception as exc:
        logger.error("Falha ao classificar pedido de recomendação (%s)", type(exc).__name__)
        return False
    return bool(resultado.get("quer_nova_recomendacao", False))
