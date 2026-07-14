"""
Schema do perfil do cidadao e a logica de extracao a partir da fala
natural da pessoa.

O perfil comeca vazio e vai sendo preenchido aos poucos, conforme a
conversa avanca. A extracao roda a cada mensagem e so atualiza os
campos que conseguiu entender -- nunca apaga o que ja estava preenchido.
"""

import json
import logging
import os
from typing import Optional

from openai import OpenAI
from pydantic import BaseModel, Field, ValidationError

from dialogue.prompts import PROMPT_EXTRACAO

logger = logging.getLogger(__name__)

CAMPOS_ESSENCIAIS = ("cidade", "escolaridade", "interesse")


class Perfil(BaseModel):
    """
    Dados minimos para recomendar um curso. So o essencial (LGPD):
    nada de CPF, nome completo ou dado sensivel sem necessidade.
    """
    cidade: Optional[str] = Field(default=None, description="Cidade ou municipio onde a pessoa mora")
    escolaridade: Optional[str] = Field(default=None, description="Etapa de escolaridade ja concluida")
    interesse: Optional[str] = Field(default=None, description="Area ou curso de interesse")
    modalidade: Optional[str] = Field(default=None, description="Presencial ou EAD, se a pessoa mencionar")

    def campos_essenciais_completos(self) -> bool:
        return all(getattr(self, campo) for campo in CAMPOS_ESSENCIAIS)

    def campos_faltantes(self) -> list[str]:
        return [c for c in CAMPOS_ESSENCIAIS if not getattr(self, c)]


def perfil_vazio() -> dict:
    """Usado na Fase 3 pra inicializar a sessao -- perfil comeca assim."""
    return Perfil().model_dump()


def determinar_fase(perfil: Perfil) -> str:
    """
    'completo' quando cidade + escolaridade + interesse estao
    preenchidos. 'modalidade' e extra, nao bloqueia.
    """
    return "completo" if perfil.campos_essenciais_completos() else "coletando"


def _chamar_llm(texto: str, perfil_atual: dict) -> dict:
    """
    Isolado numa funcao propria pra poder ser trocado/mockado nos
    testes sem precisar de chave de API de verdade.
    """
    client = OpenAI(
        api_key=os.environ["GROQ_API_KEY"],
        base_url="https://api.groq.com/openai/v1",
    )
    resposta = client.chat.completions.create(
        model=os.environ.get("GROQ_MODEL", "llama-3.3-70b-versatile"),
        messages=[
            {"role": "system", "content": PROMPT_EXTRACAO},
            {"role": "user", "content": json.dumps({
                "perfil_atual": perfil_atual,
                "mensagem": texto,
            }, ensure_ascii=False)},
        ],
        response_format={"type": "json_object"},
        temperature=0,
    )
    conteudo = resposta.choices[0].message.content
    return json.loads(conteudo)


def extrair_perfil(texto: str, perfil_atual: dict) -> Perfil:
    """
    Tenta preencher os campos faltantes do perfil a partir do que a
    pessoa disse. Nunca apaga um campo que ja estava preenchido: so
    sobrescreve quando o LLM devolve um valor novo e nao-vazio.
    """
    try:
        bruto = _chamar_llm(texto, perfil_atual)
    except Exception as exc:
        # Só o tipo da exceção -- a mensagem pode embutir uma credencial
        # vinda do cliente HTTP do Groq (ex: header de Authorization).
        logger.error("Falha ao extrair perfil via LLM (%s)", type(exc).__name__)
        return Perfil(**perfil_atual)

    mesclado = dict(perfil_atual)
    for campo in ("cidade", "escolaridade", "interesse", "modalidade"):
        valor_novo = bruto.get(campo)
        if valor_novo:
            mesclado[campo] = valor_novo

    try:
        return Perfil(**mesclado)
    except ValidationError as exc:
        logger.error("LLM devolveu perfil em formato inesperado (%s)", type(exc).__name__)
        return Perfil(**perfil_atual)