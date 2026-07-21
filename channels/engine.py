"""
Motor de resposta real, ligando o canal ao RAG.

Este arquivo é o único adaptador entre um canal (Telegram, etc) e o
motor de verdade (`retrieval.generate.answer`). Antes de chamar o RAG,
verifica se há um pedido de reinício em andamento ou recém-feito (pode
ocorrer em qualquer ponto da conversa); em seguida, se o perfil da
pessoa já está completo -- se não estiver, conduz a coleta de perfil
em vez de responder com o RAG.
"""

import json
import logging
import os
import re
from dataclasses import dataclass

import anthropic

from config.settings import settings
from dialogue.intent import precisa_busca
from dialogue.profile import OPCOES_NIVEL, Perfil, determinar_fase, extrair_perfil
from dialogue.prompts import PROMPT_COLETA, PROMPT_CONVERSA, PROMPT_CONFIRMACAO_REINICIO
from dialogue.recommendation import gerar_recomendacao, quer_nova_recomendacao
from dialogue.reset import (
    CALLBACK_REINICIO_CANCELAR,
    CALLBACK_REINICIO_CONFIRMAR,
    classificar_pedido_reinicio,
    eh_confirmacao_positiva,
    limpar_para_outra_area,
    perfil_zerado,
)
from retrieval.generate import answer

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class Botao:
    """Um botão inline: rótulo visível e o `callback_data` que volta no toque."""

    rotulo: str
    callback_data: str


class Resposta(str):
    """
    Texto de resposta que deve vir acompanhado de botões inline (hoje:
    seleção de nível, confirmação de reinício). Subclasse de `str` de
    propósito -- todo código que já trata o retorno de `responder()`
    como string simples (comparação de igualdade, f-string, `reply_text`,
    `json.dumps` no histórico) continua funcionando sem nenhuma mudança;
    só quem precisa dos botões (`channels/telegram.py`) lê `.botoes`.

    `botoes` é uma lista de linhas (cada linha uma lista de `Botao`),
    espelhando o layout de teclado do Telegram sem importar nada de
    Telegram aqui.
    """

    botoes: list[list[Botao]] | None

    def __new__(cls, texto: str, botoes: list[list[Botao]] | None = None) -> "Resposta":
        obj = super().__new__(cls, texto)
        obj.botoes = botoes
        return obj


_BOTOES_NIVEL: list[list[Botao]] = [
    [Botao(rotulo, f"nivel:{i}") for i, (rotulo, _) in enumerate(OPCOES_NIVEL)][0:2],
    [Botao(rotulo, f"nivel:{i}") for i, (rotulo, _) in enumerate(OPCOES_NIVEL)][2:4],
]

# Opção segura (nao-destrutiva) primeiro, e rotulos que restatam a
# consequencia real -- nunca "Sim"/"Nao" vago -- seguindo a diretriz de
# UX pra acoes destrutivas: friccao e clareza proporcionais ao risco.
_BOTOES_REINICIO: list[list[Botao]] = [
    [Botao("Manter meus dados", CALLBACK_REINICIO_CANCELAR)],
    [Botao("Apagar tudo e recomeçar", CALLBACK_REINICIO_CONFIRMAR)],
]

_MENSAGEM_FALLBACK = (
    "Desculpa, tive um problema pra buscar essa informação agora. "
    "Tenta de novo em instantes, por favor."
)

_MENSAGEM_FALLBACK_CONFIRMACAO_REINICIO = (
    "Quer mesmo apagar tudo e começar de novo? Responde 'sim' pra confirmar."
)

_EXTENSOES = (".pdf", ".docx", ".doc", ".odt", ".pptx")

# Teto de tamanho de mensagem: protege qualquer canal que chame
# responder() (não só o Telegram) contra abuso via mensagem gigante
# nos motores pagos (Anthropic/Voyage).
_MAX_CARACTERES_MENSAGEM = 4000


def _logar_falha(operacao: str, user_id: str, exc: Exception) -> None:
    """
    Loga só o tipo da exceção, nunca sua mensagem nem o traceback: uma
    lib downstream (cliente HTTP do Anthropic/Voyage) pode embutir uma
    credencial na mensagem de erro, e `logger.exception` gravaria isso
    sem filtro no log de aplicação.
    """
    logger.error("Falha ao %s para user_id=%s (%s)", operacao, user_id, type(exc).__name__)


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
    client = anthropic.Anthropic()
    contexto = {
        "perfil_atual": perfil.model_dump(),
        "campos_faltantes": perfil.campos_faltantes(),
    }
    resposta = client.messages.create(
        model=settings.anthropic_model_geracao,
        max_tokens=500,
        system=PROMPT_COLETA,
        messages=[{"role": "user", "content": json.dumps(contexto, ensure_ascii=False)}],
    )
    return next(b.text for b in resposta.content if b.type == "text")


def _gerar_resposta_conversa(texto: str) -> str:
    """
    Usa o LLM pra responder papo informal (saudação, agradecimento,
    pergunta sobre o próprio bot) sem rodar retrieval nem citar fonte
    -- roteado aqui por `dialogue.intent.precisa_busca` antes de
    chegar no RAG.
    """
    client = anthropic.Anthropic()
    resposta = client.messages.create(
        model=settings.anthropic_model_geracao,
        max_tokens=200,
        system=PROMPT_CONVERSA,
        messages=[{"role": "user", "content": texto}],
    )
    return next(b.text for b in resposta.content if b.type == "text")


def _gerar_confirmacao_reinicio(texto: str) -> str:
    """
    Usa o LLM pra formular a pergunta de confirmação de reinício,
    de forma acolhedora, com base no que a pessoa pediu.
    """
    client = anthropic.Anthropic()
    resposta = client.messages.create(
        model=settings.anthropic_model_geracao,
        max_tokens=100,
        system=PROMPT_CONFIRMACAO_REINICIO,
        messages=[{"role": "user", "content": texto}],
    )
    return next(b.text for b in resposta.content if b.type == "text")


def _com_botoes_de_nivel(pergunta: str, perfil: Perfil) -> str | Resposta:
    """
    Envolve a pergunta de coleta com os botões de nível quando "nivel"
    é o próximo campo faltante -- único ponto em que decide se anexa
    esses botões, usado nos dois lugares que chamam
    `_gerar_pergunta_coleta` (coleta normal e pós-"buscar outra área").
    """
    faltantes = perfil.campos_faltantes()
    if faltantes and faltantes[0] == "nivel":
        return Resposta(pergunta, botoes=_BOTOES_NIVEL)
    return pergunta


def responder(
    user_id: str,
    texto: str,
    sessao: dict,
    nivel_escolhido: str | None = None,
) -> str:
    """
    Recebe o id do usuário, o texto que ele mandou, e a sessão atual.

    `nivel_escolhido` é preenchido só quando a origem da chamada foi um
    toque num botão inline de nível (`channels/telegram.py`): pula a
    chamada ao extrator (LLM) e define `Perfil.nivel` direto com o
    valor do botão -- mais barato e determinístico que tratar o rótulo
    do botão como se a pessoa tivesse digitado. Por construção, um
    botão de nível só existe durante a fase "coletando", nunca pode ser
    um pedido de reinício, então também pula os blocos de reinício
    abaixo (`classificar_pedido_reinicio` custaria uma chamada paga à
    toa nesse caso).

    Antes de qualquer outra coisa, verifica se há um pedido de
    reinício em andamento (`fase_dialogo == "confirmando_reinicio"`)
    ou recém-feito nesta mensagem (`classificar_pedido_reinicio`), já
    que isso pode ser pedido a qualquer momento da conversa,
    independente da fase de coleta/recomendação/RAG. Ao entrar em
    "confirmando_reinicio", a fase anterior é guardada em
    `sessao["fase_dialogo_anterior"]`, pra poder ser restaurada caso a
    pessoa recue da confirmação -- sem isso, uma recusa jogaria
    incorretamente qualquer perfil incompleto direto pra fase
    "completo".

    Se o perfil ainda não estiver completo, extrai o que der da
    mensagem (com o histórico recente como contexto pra resolver
    referências tipo "e advogado?"), atualiza a sessão (in-place --
    quem chama esta função é responsável por persistir a sessão de
    volta no Redis) e devolve a próxima pergunta de coleta. Quando o
    perfil acaba de ficar completo neste turno, devolve a recomendação
    do motor estruturado (`recommend/opportunities.py`). Com o perfil
    já completo de antes, uma nova recomendação só é gerada se a
    pessoa pedir explicitamente (`quer_nova_recomendacao`); nesse caso,
    a mensagem atual passa de novo pelo extrator antes de recomendar,
    pra capturar interesse/modalidade diferente do que já estava salvo
    (`interesse` é sugestão, não filtro rígido -- pode mudar a cada
    pedido). Sem pedido de recomendação, `precisa_busca` decide entre
    RAG (pergunta real sobre edital) e uma resposta simples de papo
    informal (saudação, agradecimento, pergunta sobre o próprio bot),
    sem gastar retrieval nem citar fonte nessa segunda opção.
    """
    texto = texto[:_MAX_CARACTERES_MENSAGEM]

    if nivel_escolhido is None:
        # Reinício: verificado antes de qualquer outra coisa, pois pode
        # ser pedido em qualquer ponto da conversa.
        if sessao.get("fase_dialogo") == "confirmando_reinicio":
            if eh_confirmacao_positiva(texto):
                sessao["perfil"] = perfil_zerado()
                sessao["fase_dialogo"] = "coletando"
                sessao["historico"] = []
                sessao.pop("fase_dialogo_anterior", None)
                return "Prontinho, apaguei tudo! Vamos começar de novo: em qual cidade você mora?"
            else:
                # Restaura a fase em que a pessoa estava antes de pedir o
                # reinício (pode ter sido "coletando" ou "completo"). O
                # fallback "completo" só entra em cena defensivamente, se
                # por algum motivo a sessão chegar aqui sem o campo salvo
                # (ex.: sessão antiga no Redis, de antes desse campo
                # existir).
                sessao["fase_dialogo"] = sessao.pop("fase_dialogo_anterior", "completo")
                return "Sem problema, mantive seus dados como estavam."

        # `any(...)` em vez de só checar a chave: um perfil recem-criado
        # (`perfil_vazio()`) já é um dict não-vazio, só que com todos os
        # campos None -- sem essa checagem, classificar_pedido_reinicio()
        # rodaria (Anthropic pago) em toda mensagem desde o segundo turno,
        # mesmo antes de existir qualquer dado real pra reiniciar.
        if any((sessao.get("perfil") or {}).values()):
            pedido = classificar_pedido_reinicio(texto)
            if pedido == "buscar_outra_area":
                sessao["perfil"] = limpar_para_outra_area(sessao["perfil"])
                perfil_pos_reset = Perfil(**sessao["perfil"])
                sessao["fase_dialogo"] = determinar_fase(perfil_pos_reset)
                try:
                    pergunta = _gerar_pergunta_coleta(perfil_pos_reset)
                except Exception as exc:
                    _logar_falha("gerar pergunta apos buscar outra area", user_id, exc)
                    return _MENSAGEM_FALLBACK
                return _com_botoes_de_nivel(pergunta, perfil_pos_reset)
            elif pedido == "comecar_de_novo":
                sessao["fase_dialogo_anterior"] = sessao.get("fase_dialogo")
                sessao["fase_dialogo"] = "confirmando_reinicio"
                try:
                    pergunta = _gerar_confirmacao_reinicio(texto)
                except Exception as exc:
                    _logar_falha("gerar confirmacao de reinicio", user_id, exc)
                    return Resposta(_MENSAGEM_FALLBACK_CONFIRMACAO_REINICIO, botoes=_BOTOES_REINICIO)
                return Resposta(pergunta, botoes=_BOTOES_REINICIO)

    perfil_atual = sessao.get("perfil") or {}
    perfil = Perfil(**perfil_atual)

    if determinar_fase(perfil) != "completo":
        historico = sessao.get("historico") or []
        if nivel_escolhido:
            perfil = Perfil(**{**perfil_atual, "nivel": nivel_escolhido})
        else:
            perfil = extrair_perfil(texto, perfil_atual, historico=historico)
        sessao["perfil"] = perfil.model_dump()
        sessao["fase_dialogo"] = determinar_fase(perfil)

        if sessao["fase_dialogo"] != "completo":
            try:
                pergunta = _gerar_pergunta_coleta(perfil)
            except Exception as exc:
                _logar_falha("gerar pergunta de coleta", user_id, exc)
                return _MENSAGEM_FALLBACK
            return _com_botoes_de_nivel(pergunta, perfil)

        try:
            return gerar_recomendacao(perfil)
        except Exception as exc:
            _logar_falha("gerar recomendação", user_id, exc)
            return _MENSAGEM_FALLBACK

    if quer_nova_recomendacao(texto):
        try:
            historico = sessao.get("historico") or []
            perfil = extrair_perfil(texto, perfil.model_dump(), historico=historico)
            sessao["perfil"] = perfil.model_dump()
            return gerar_recomendacao(perfil)
        except Exception as exc:
            _logar_falha("gerar nova recomendação", user_id, exc)
            return _MENSAGEM_FALLBACK

    if not precisa_busca(texto):
        try:
            return _gerar_resposta_conversa(texto)
        except Exception as exc:
            _logar_falha("gerar resposta de conversa", user_id, exc)
            return _MENSAGEM_FALLBACK

    try:
        resultado = answer(texto)
    except Exception as exc:
        _logar_falha("chamar answer()", user_id, exc)
        return _MENSAGEM_FALLBACK

    texto_resposta = (resultado or {}).get("answer")
    if not texto_resposta:
        logger.error("answer() devolveu resposta vazia para user_id=%s", user_id)
        return _MENSAGEM_FALLBACK

    sources = (resultado or {}).get("sources")
    if sources:
        return f"{texto_resposta}\n\n{_formatar_fontes(sources)}"

    return texto_resposta