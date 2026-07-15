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
from datetime import date

import anthropic
from pydantic import BaseModel

from config.settings import settings
from dialogue.profile import Perfil
from dialogue.prompts import PROMPT_CLASSIFICA_PEDIDO_RECOMENDACAO, PROMPT_RECOMENDACAO
from recommend.opportunities import recomendar

logger = logging.getLogger(__name__)


class _ClassificacaoPedido(BaseModel):
    quer_nova_recomendacao: bool


def montar_contexto(perfil: Perfil, hoje: date) -> dict:
    """
    Chama o motor de recomendacao e serializa o resultado num dict
    JSON-safe (datas viram string ISO) pronto pra entrar no prompt.

    `recomendar()` agrupa por camada de proximidade (`na_cidade`,
    `regiao`, `ead`, `outras_cidades`) em vez de filtrar cidade como
    fronteira rigida -- repassamos as camadas como vieram, pra redacao
    poder ser transparente sobre o quanto de deslocamento cada opcao
    implica. Sem `alcance` explicito, usa o default de `recomendar()`
    (inclusivo, mas nunca extrapola pra `outras_cidades` sozinho).
    """
    resultado = recomendar(
        cidade=perfil.cidade, hoje=hoje, nivel=perfil.nivel, modalidade=perfil.modalidade
    )
    return {
        "interesse": perfil.interesse,
        "na_cidade": [o.model_dump(mode="json") for o in resultado["na_cidade"]],
        "regiao": [o.model_dump(mode="json") for o in resultado["regiao"]],
        "ead": [o.model_dump(mode="json") for o in resultado["ead"]],
        "outras_cidades": [o.model_dump(mode="json") for o in resultado["outras_cidades"]],
        "proxima": resultado["proxima"].model_dump(mode="json") if resultado["proxima"] else None,
    }


def _chamar_llm(contexto: dict) -> str:
    """
    Isolado numa funcao propria pra poder ser trocado/mockado nos
    testes sem precisar de chave de API de verdade.
    """
    client = anthropic.Anthropic()
    resposta = client.messages.create(
        model=settings.anthropic_model_geracao,
        max_tokens=1200,
        system=PROMPT_RECOMENDACAO,
        messages=[
            {
                "role": "user",
                "content": (
                    "Contexto (JSON, já calculado, é a ÚNICA fonte de verdade): "
                    + json.dumps(contexto, ensure_ascii=False)
                ),
            }
        ],
    )
    return next(b.text for b in resposta.content if b.type == "text")


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
    client = anthropic.Anthropic()
    resposta = client.messages.parse(
        model=settings.anthropic_model_extracao,
        max_tokens=256,
        system=PROMPT_CLASSIFICA_PEDIDO_RECOMENDACAO,
        messages=[{"role": "user", "content": texto}],
        output_format=_ClassificacaoPedido,
    )
    return resposta.parsed_output.model_dump()


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
