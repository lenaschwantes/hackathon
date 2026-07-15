"""
Schema do perfil do cidadao e a logica de extracao a partir da fala
natural da pessoa.

O perfil comeca vazio e vai sendo preenchido aos poucos, conforme a
conversa avanca. A extracao roda a cada mensagem e so atualiza os
campos que conseguiu entender -- nunca apaga o que ja estava preenchido.
"""

import json
import logging
from typing import Optional

import anthropic
from pydantic import BaseModel, Field, ValidationError

from config.settings import settings
from dialogue.prompts import PROMPT_EXTRACAO

logger = logging.getLogger(__name__)

CAMPOS_ESSENCIAIS = ("cidade", "escolaridade", "interesse", "nivel")
_CAMPOS_EXTRAIDOS = (*CAMPOS_ESSENCIAIS, "modalidade")

# Máximo de turnos recentes passados como contexto pra extração --
# só o suficiente pra resolver uma referência à mensagem anterior, sem
# inflar o prompt.
_MAX_HISTORICO_NO_PROMPT = 4


class Perfil(BaseModel):
    """
    Dados minimos para recomendar um curso. So o essencial (LGPD):
    nada de CPF, nome completo ou dado sensivel sem necessidade.
    """
    cidade: Optional[str] = Field(default=None, description="Cidade ou municipio onde a pessoa mora")
    escolaridade: Optional[str] = Field(default=None, description="Etapa de escolaridade ja concluida")
    interesse: Optional[str] = Field(default=None, description="Area ou curso de interesse")
    nivel: Optional[str] = Field(
        default=None,
        description="Nivel de curso desejado: tecnico integrado, tecnico subsequente, superior ou FIC",
    )
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
    'completo' quando cidade + escolaridade + interesse + nivel estao
    preenchidos. 'modalidade' e extra, nao bloqueia.
    """
    return "completo" if perfil.campos_essenciais_completos() else "coletando"


def _chamar_llm(texto: str, perfil_atual: dict, historico: list[dict] | None = None) -> dict:
    """
    Isolado numa funcao propria pra poder ser trocado/mockado nos
    testes sem precisar de chave de API de verdade. Usa o proprio
    `Perfil` como schema de output estruturado -- os campos batem
    exatamente com o que precisa ser extraido.
    """
    client = anthropic.Anthropic()
    payload = {
        "perfil_atual": perfil_atual,
        "mensagem": texto,
    }
    if historico:
        payload["historico"] = historico[-_MAX_HISTORICO_NO_PROMPT:]

    resposta = client.messages.parse(
        model=settings.anthropic_model_extracao,
        max_tokens=512,
        system=PROMPT_EXTRACAO,
        messages=[{"role": "user", "content": json.dumps(payload, ensure_ascii=False)}],
        output_format=Perfil,
    )
    return resposta.parsed_output.model_dump()


def extrair_perfil(texto: str, perfil_atual: dict, historico: list[dict] | None = None) -> Perfil:
    """
    Tenta preencher os campos faltantes do perfil a partir do que a
    pessoa disse. Nunca apaga um campo que ja estava preenchido: so
    sobrescreve quando o LLM devolve um valor novo e nao-vazio.

    `historico` (opcional, ultimas mensagens da conversa) da contexto
    pra resolver referencia a pergunta anterior -- ex: se a ultima
    pergunta do bot foi sobre interesse e a pessoa responde so
    "advogado", o LLM entende que isso preenche "interesse".
    """
    try:
        bruto = _chamar_llm(texto, perfil_atual, historico)
    except Exception as exc:
        # Só o tipo da exceção -- a mensagem pode embutir uma credencial
        # vinda do cliente HTTP da Anthropic (ex: header de Authorization).
        logger.error("Falha ao extrair perfil via LLM (%s)", type(exc).__name__)
        return Perfil(**perfil_atual)

    mesclado = dict(perfil_atual)
    for campo in _CAMPOS_EXTRAIDOS:
        valor_novo = bruto.get(campo)
        if valor_novo:
            mesclado[campo] = valor_novo

    try:
        return Perfil(**mesclado)
    except ValidationError as exc:
        logger.error("LLM devolveu perfil em formato inesperado (%s)", type(exc).__name__)
        return Perfil(**perfil_atual)