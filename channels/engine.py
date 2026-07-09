"""
Motor de resposta real, ligando o canal ao RAG.

Este arquivo é o único adaptador entre um canal (Telegram, etc) e o
motor de verdade (`retrieval.generate.answer`). Nenhuma lógica de
negócio de RAG mora aqui — só a formatação da resposta e a proteção
contra erros vazando pro usuário final.
"""

import logging

from retrieval.generate import answer

logger = logging.getLogger(__name__)

_MENSAGEM_FALLBACK = (
    "Desculpa, tive um problema pra buscar essa informação agora. "
    "Tenta de novo em instantes, por favor."
)


def responder(user_id: str, texto: str, sessao: dict) -> str:
    """
    Recebe o id do usuário, o texto que ele mandou, e a sessão atual
    (o "estado" da conversa dele, vindo do Redis).

    Chama o RAG de verdade (`retrieval.generate.answer`) e formata a
    resposta pro canal. `sessao` ainda não é usada aqui — a
    personalização por perfil é Fase 2 — mas o parâmetro fica na
    assinatura pra não quebrar o contrato quando ela chegar.
    """
    try:
        resultado = answer(texto)
    except Exception:
        logger.exception("Falha ao chamar answer() para user_id=%s", user_id)
        return _MENSAGEM_FALLBACK

    texto_resposta = (resultado or {}).get("answer")
    if not texto_resposta:
        logger.error("answer() devolveu resposta vazia para user_id=%s", user_id)
        return _MENSAGEM_FALLBACK

    sources = (resultado or {}).get("sources")
    if sources:
        editais = ", ".join(sources)
        return f"{texto_resposta}\n\nFontes: {editais}"

    return texto_resposta
