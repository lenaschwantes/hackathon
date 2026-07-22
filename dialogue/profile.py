"""
Schema do perfil do cidadao e a logica de extracao a partir da fala
natural da pessoa.

O perfil comeca vazio e vai sendo preenchido aos poucos, conforme a
conversa avanca. A extracao roda a cada mensagem e so atualiza os
campos que conseguiu entender -- nunca apaga o que ja estava preenchido.
"""

import json
import logging
import re
import unicodedata
from typing import Optional

import anthropic
from pydantic import BaseModel, Field, ValidationError

from config.prompts import PROMPT_EXTRACAO
from config.settings import settings

logger = logging.getLogger(__name__)

CAMPOS_ESSENCIAIS = ("cidade", "escolaridade", "interesse", "nivel")
_CAMPOS_EXTRAIDOS = (*CAMPOS_ESSENCIAIS, "modalidade", "alcance")

# Ordem em que os campos sao perguntados na coleta. Igual a
# CAMPOS_ESSENCIAIS, mas com "alcance" intercalado antes de "nivel" --
# ele e extra e nao entra em `campos_essenciais_completos()`, mas
# intercalado antes do ultimo campo essencial ele ganha uma pergunta
# dedicada no fluxo normal (turno a turno), antes do perfil fechar.
# Se os essenciais chegarem todos de uma vez (num so turno), o perfil
# fecha sem alcance ter sido perguntado -- normal, ver
# `dialogue/recommendation.py` pro default nesse caso.
_ORDEM_COLETA = ("cidade", "escolaridade", "interesse", "alcance", "nivel")

# Opcoes fechadas do campo "escolaridade", na ordem em que devem ser
# apresentadas (rotulo legivel, valor canonico). Os valores batem
# exatamente com o vocabulario fechado que PROMPT_EXTRACAO ja garante
# extrair de fala livre -- usado pelos botoes inline do Telegram
# (`channels/engine.py`), que pulam o extrator e usam o valor direto
# quando a pessoa toca um botao.
OPCOES_ESCOLARIDADE: tuple[tuple[str, str], ...] = (
    ("Ensino fundamental", "ensino fundamental"),
    ("Ensino médio", "ensino medio"),
    ("Ensino médio técnico", "ensino medio tecnico"),
    ("Já fiz uma faculdade", "superior"),
)

# Opcoes fechadas do campo "alcance", na ordem em que devem ser
# apresentadas (rotulo legivel, valor canonico). Os valores batem
# exatamente com `_ALCANCES_VALIDOS` (`dialogue/recommendation.py`) e
# com o vocabulario fechado que PROMPT_EXTRACAO ja garante extrair de
# fala livre -- usado pelos botoes inline do Telegram
# (`channels/engine.py`), que pulam o extrator e usam o valor direto
# quando a pessoa toca um botao.
OPCOES_ALCANCE: tuple[tuple[str, str], ...] = (
    ("Só na minha cidade", "local"),
    ("Cidade próxima também", "regional"),
    ("Prefiro a distância (EAD)", "ead"),
    ("Não me importo com o lugar", "qualquer"),
)

# Opcoes fechadas do campo "nivel", na ordem em que devem ser
# apresentadas (rotulo legivel, valor canonico). Os valores batem
# exatamente com o que PROMPT_EXTRACAO ja garante extrair de fala
# livre -- usado pelos botoes inline do Telegram (`channels/engine.py`),
# que pulam o extrator e usam o valor direto quando a pessoa toca um
# botao.
OPCOES_NIVEL: tuple[tuple[str, str], ...] = (
    ("Técnico integrado", "tecnico integrado"),
    ("Técnico subsequente", "tecnico subsequente"),
    ("Graduação", "superior"),
    ("FIC (curso rápido)", "FIC"),
)

_TODOS_OS_NIVEIS: tuple[str, ...] = tuple(valor for _, valor in OPCOES_NIVEL)

# Níveis coerentes com cada escolaridade já concluída -- evita oferecer
# "tecnico integrado" (pensado pra quem ainda vai cursar o ensino medio,
# junto com ele) pra quem ja tem superior, ou "superior" pra quem so tem
# o fundamental. FIC (curso rapido de qualificacao) nao exige etapa
# anterior especifica, entao continua compativel com todas.
_NIVEIS_POR_ESCOLARIDADE: dict[str, tuple[str, ...]] = {
    "ensino fundamental": ("tecnico integrado", "FIC"),
    "ensino medio": ("tecnico subsequente", "superior", "FIC"),
    "ensino medio completo": ("tecnico subsequente", "superior", "FIC"),
    "ensino medio tecnico": ("tecnico subsequente", "superior", "FIC"),
    "superior": ("FIC",),
    "superior completo": ("FIC",),
}


def niveis_compativeis(escolaridade: str | None) -> tuple[str, ...]:
    """Níveis de curso coerentes com a escolaridade já concluída.

    Sem escolaridade (ainda) coletada, ou com um valor fora do
    vocabulário fechado que `PROMPT_EXTRACAO` garante, nada é
    restringido -- devolve todos os níveis, na ordem de `OPCOES_NIVEL`.
    Compara normalizado (`_normaliza`) como rede de segurança contra
    variação de acento/caixa ou um "completo" a mais que o LLM devolva
    por conta própria, apesar do vocabulário fechado pedido no prompt.
    """
    if not escolaridade:
        return _TODOS_OS_NIVEIS
    return _NIVEIS_POR_ESCOLARIDADE.get(_normaliza(escolaridade), _TODOS_OS_NIVEIS)


def aplicar_coerencia_nivel(perfil_bruto: dict) -> dict:
    """Preenche "nivel" quando a escolaridade já deixa só um nível
    plausível (ex.: "superior" só combina com "FIC") -- evita perguntar
    de novo um campo que já está implícito. Com mais de uma opção
    compatível, deixa "nivel" faltante mesmo: quem pergunta
    (`_gerar_pergunta_coleta`/`_com_botoes_de_campo_fechado`, em
    `channels/engine.py`) restringe as opções via `niveis_compativeis`,
    sem perguntar a toa.

    Usada tanto por `extrair_perfil` (fala livre) quanto pelo bypass de
    botão de escolaridade em `channels/engine.py` (que pula o extrator
    mas precisa da mesma coerência).
    """
    if perfil_bruto.get("nivel") or not perfil_bruto.get("escolaridade"):
        return perfil_bruto
    compativeis = niveis_compativeis(perfil_bruto["escolaridade"])
    if len(compativeis) == 1:
        return {**perfil_bruto, "nivel": compativeis[0]}
    return perfil_bruto

# Máximo de turnos recentes passados como contexto pra extração --
# só o suficiente pra resolver uma referência à mensagem anterior, sem
# inflar o prompt.
_MAX_HISTORICO_NO_PROMPT = 4

# Valor aceito pra "interesse" quando a pessoa insiste que nao sabe --
# nao-vazio de proposito, pra satisfazer campos_essenciais_completos()
# sem precisar de nenhum ajuste ali. Ver _eh_resposta_nao_sei().
_INTERESSE_SEM_PREFERENCIA = "sem preferência definida"

_RESPOSTAS_NAO_SEI = frozenset(
    {
        "nao sei", "ainda nao sei", "num sei", "sei nao", "nao sei bem",
        "nao sei ainda", "nao faco ideia", "sem ideia", "nao tenho ideia",
        "nao tenho certeza", "qualquer coisa", "tanto faz",
    }
)

_PONTUACAO_FINAL = re.compile(r"[!.,;?]+$")


def _normaliza(texto: str) -> str:
    """Normaliza texto para comparação: minúsculo, sem acento, sem pontuação final."""
    sem_acento = unicodedata.normalize("NFKD", texto).encode("ascii", "ignore").decode()
    return _PONTUACAO_FINAL.sub("", sem_acento.strip().lower()).strip()


def _eh_resposta_nao_sei(texto: str) -> bool:
    """Reconhece uma resposta de 'nao sei' (ou variacao) pro campo de interesse."""
    return _normaliza(texto) in _RESPOSTAS_NAO_SEI


def _ultima_mensagem_do_usuario(historico: list[dict] | None) -> str | None:
    """Devolve o texto da ultima mensagem marcada 'de': 'usuario' no historico, se houver."""
    if not historico:
        return None
    for turno in reversed(historico):
        if turno.get("de") == "usuario":
            return turno.get("texto")
    return None


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
    alcance: Optional[str] = Field(
        default=None,
        description=(
            "Alcance geografico que a pessoa aceita pra estudar: "
            "'local' (so a propria cidade), 'regional' (topa se "
            "deslocar pra cidade vizinha), 'ead' (prefere ou so "
            "consegue a distancia) ou 'qualquer' (nao se importa com "
            "o lugar)."
        ),
    )

    def campos_essenciais_completos(self) -> bool:
        return all(getattr(self, campo) for campo in CAMPOS_ESSENCIAIS)

    def campos_faltantes(self) -> list[str]:
        return [c for c in _ORDEM_COLETA if not getattr(self, c)]


def perfil_vazio() -> dict:
    """Usado na Fase 3 pra inicializar a sessao -- perfil comeca assim."""
    return Perfil().model_dump()


def determinar_fase(perfil: Perfil) -> str:
    """
    'completo' quando cidade + escolaridade + interesse + nivel estao
    preenchidos. 'modalidade' e 'alcance' sao extra, nao bloqueiam.
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

    # "Nao sei" pro campo de interesse: na primeira vez, nao preenche
    # nada (o LLM corretamente devolveu null acima) -- o bot reformula
    # pedindo exemplos, comportamento normal do PROMPT_COLETA. So na
    # SEGUNDA vez seguida que a pessoa insiste (mensagem atual e a
    # anterior dela no historico sao ambas "nao sei") e que aceita e
    # avanca sem area especifica -- sem isso, alguem que genuinamente
    # nao sabe fica preso pra sempre na mesma pergunta, ja que
    # "interesse" e campo essencial.
    if not mesclado.get("interesse") and _eh_resposta_nao_sei(texto):
        anterior = _ultima_mensagem_do_usuario(historico)
        if anterior is not None and _eh_resposta_nao_sei(anterior):
            mesclado["interesse"] = _INTERESSE_SEM_PREFERENCIA

    mesclado = aplicar_coerencia_nivel(mesclado)

    try:
        return Perfil(**mesclado)
    except ValidationError as exc:
        logger.error("LLM devolveu perfil em formato inesperado (%s)", type(exc).__name__)
        return Perfil(**perfil_atual)