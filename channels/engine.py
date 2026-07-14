"""
Motor de resposta real, ligando o canal ao RAG.

Este arquivo é o único adaptador entre um canal (Telegram, etc) e o
motor de verdade (`retrieval.generate.answer`). Antes de chamar o RAG,
verifica se o perfil da pessoa já está completo -- se não estiver,
conduz a coleta de perfil em vez de responder com o RAG.
"""

import logging
import os
import re

from openai import OpenAI

from dialogue.profile import Perfil, determinar_fase, extrair_perfil
from dialogue.prompts import PROMPT_COLETA
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


def _gerar_pergunta_coleta(perfil: Perfil) -> str:
    """
    Usa o LLM pra formular a próxima pergunta de coleta de forma
    acolhedora, com base no que já se sabe e no que ainda falta.
    """
    client = OpenAI(
        api_key=os.environ["GROQ_API_KEY"],
        base_url="https://api.groq.com/openai/v1",
    )
    prompt = PROMPT_COLETA.format(
        perfil_atual=perfil.model_dump(),
        campos_faltantes=perfil.campos_faltantes(),
    )
    resposta = client.chat.completions.create(
        model=os.environ.get("GROQ_MODEL", "llama-3.3-70b-versatile"),
        messages=[{"role": "system", "content": prompt}],
        temperature=0.7,
    )
    return resposta.choices[0].message.content


def responder(user_id: str, texto: str, sessao: dict) -> str:
    """
    Recebe o id do usuário, o texto que ele mandou, e a sessão atual.

    Se o perfil ainda não estiver completo, extrai o que der da
    mensagem, atualiza a sessão (in-place -- quem chama esta função
    é responsável por persistir a sessão de volta no Redis) e
    devolve a próxima pergunta de coleta. Só chama o RAG quando o
    perfil já estiver completo.
    """
    perfil_atual = sessao.get("perfil") or {}
    perfil = Perfil(**perfil_atual)

    if determinar_fase(perfil) != "completo":
        perfil = extrair_perfil(texto, perfil_atual)
        sessao["perfil"] = perfil.model_dump()
        sessao["fase_dialogo"] = determinar_fase(perfil)

        if sessao["fase_dialogo"] != "completo":
            try:
                return _gerar_pergunta_coleta(perfil)
            except Exception:
                logger.exception("Falha ao gerar pergunta de coleta para user_id=%s", user_id)
                return _MENSAGEM_FALLBACK
        # se acabou de completar, cai direto pro RAG abaixo, na mesma resposta

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