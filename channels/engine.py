"""
Motor de resposta real, ligando o canal ao RAG.

Este arquivo é o único adaptador entre um canal (Telegram, etc) e o
motor de verdade (`retrieval.generate.answer`). Nenhuma lógica de
negócio de RAG mora aqui — só a formatação da resposta e a proteção
contra erros vazando pro usuário final.
"""

import logging
import os
import re

from retrieval.generate import answer

logger = logging.getLogger(__name__)

_MENSAGEM_FALLBACK = (
    "Desculpa, tive um problema pra buscar essa informação agora. "
    "Tenta de novo em instantes, por favor."
)

_EXTENSOES = (".pdf", ".docx", ".doc", ".odt", ".pptx")


def _rotulo_fonte(file_name: str) -> str:
    """Deriva um rótulo legível a partir do nome de arquivo bruto.

    Nomes vindos do crawler (título real da página do IFSC) já são
    legíveis — só a extensão é removida. Nomes "crus" (com `_`, típicos
    de slug de URL ou de arquivo colocado manualmente) passam por uma
    normalização adicional.

    Ex.: "EDITAL 05_2026_2-Cadastro-de-Reserva-ok.odt.pdf"
         -> "Edital 05/2026 - Cadastro de Reserva"
    """
    nome = file_name
    raiz, ext = os.path.splitext(nome)
    while ext.lower() in _EXTENSOES:
        nome = raiz
        raiz, ext = os.path.splitext(nome)

    if "_" in nome:
        # "05_2026_2-" (número_ano_semestre seguido de separador) -> "05/2026 - "
        # o marcador provisório protege esse traço do collapse genérico abaixo
        nome = re.sub(
            r"(\d+)_(\d{4})(?:_\d+)?(-)?",
            lambda m: f"{m.group(1)}/{m.group(2)}" + (" \x00DASH\x00 " if m.group(3) else ""),
            nome,
        )
        nome = re.sub(r"[_-]+", " ", nome)
        nome = nome.replace("\x00DASH\x00", "-")
        nome = re.sub(r"\bok\b", "", nome, flags=re.IGNORECASE)
        nome = re.sub(r"\s+", " ", nome).strip()

    nome = re.sub(r"^EDITAL\b", "Edital", nome)
    return nome or file_name


def _formatar_fontes(sources: list[str]) -> str:
    rotulos = [_rotulo_fonte(s) for s in sources]
    rotulo_campo = "Fonte" if len(rotulos) == 1 else "Fontes"
    return f"{rotulo_campo}: {', '.join(rotulos)}"


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
        return f"{texto_resposta}\n\n{_formatar_fontes(sources)}"

    return texto_resposta
