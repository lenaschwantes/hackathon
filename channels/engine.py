"""
Motor de resposta real, ligando o canal ao RAG.

Este arquivo é o único adaptador entre um canal (Telegram, etc) e o
motor de verdade (`retrieval.generate.answer`). Antes de chamar o RAG,
verifica se há um pedido de reinício em andamento ou recém-feito (pode
ocorrer em qualquer ponto da conversa); numa sessão nova, resolve
primeiro a bifurcação inicial (buscar curso vs. só tirar uma dúvida,
sem obrigar coleta de perfil pra isso); em seguida, se o perfil da
pessoa já está completo -- se não estiver, conduz a coleta de perfil
em vez de responder com o RAG.
"""

import json
import logging
import os
import re
from dataclasses import dataclass

import anthropic

from config.prompts import PROMPT_COLETA, PROMPT_CONVERSA, PROMPT_CONFIRMACAO_REINICIO
from config.settings import settings
from dialogue.intent import precisa_busca
from dialogue.editais_catalogo import buscar_edital_por_indice, carregar_editais_abertos
from dialogue.onboarding import (
    CALLBACK_INICIO_BUSCAR,
    CALLBACK_INICIO_DUVIDA,
    CALLBACK_DUVIDA_GUIA_CURSOS,
    CALLBACK_DUVIDA_PRAZOS,
    CALLBACK_DUVIDA_PERGUNTA_LIVRE,
    CALLBACK_EDITAL_VER_OUTRO,
    CALLBACK_EDITAL_ENCERRAR,
    TEXTO_SINTETICO_BUSCAR_CURSO,
    TEXTO_SINTETICO_TENHO_DUVIDA,
    TEXTO_SINTETICO_GUIA_CURSOS,
    TEXTO_SINTETICO_DUVIDA_PRAZOS,
    TEXTO_SINTETICO_PERGUNTA_LIVRE,
    TEXTO_SINTETICO_VER_OUTRO_EDITAL,
    TEXTO_SINTETICO_ENCERRAR_DUVIDA,
)
from dialogue.profile import (
    OPCOES_ALCANCE,
    OPCOES_ESCOLARIDADE,
    OPCOES_NIVEL,
    Perfil,
    aplicar_coerencia_nivel,
    determinar_fase,
    extrair_perfil,
    niveis_compativeis,
)
from dialogue.recommendation import gerar_recomendacao, quer_nova_recomendacao
from dialogue.reset import (
    CALLBACK_REINICIO_CANCELAR,
    CALLBACK_REINICIO_CONFIRMAR,
    classificar_pedido_reinicio,
    eh_confirmacao_positiva,
    eh_gatilho_explicito_de_reinicio_total,
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
    seleção de escolaridade, seleção de nível, confirmação de
    reinício). Subclasse de `str` de
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


def _botoes_nivel(escolaridade: str | None) -> list[list[Botao]]:
    """Botões de nível, restritos aos coerentes com a escolaridade já
    coletada (`dialogue.profile.niveis_compativeis`) -- ex.: quem já fez
    faculdade não vê mais "Técnico integrado" entre as opções.

    `callback_data` usa o índice original em `OPCOES_NIVEL`, não a
    posição na lista filtrada, pra `channels/telegram.py` continuar
    resolvendo `OPCOES_NIVEL[i]` certo quando a pessoa toca um botão.
    """
    compativeis = niveis_compativeis(escolaridade)
    botoes = [
        Botao(rotulo, f"nivel:{i}")
        for i, (rotulo, valor) in enumerate(OPCOES_NIVEL)
        if valor in compativeis
    ]
    return [botoes[i : i + 2] for i in range(0, len(botoes), 2)]


_BOTOES_ESCOLARIDADE: list[list[Botao]] = [
    [Botao(rotulo, f"escolaridade:{i}") for i, (rotulo, _) in enumerate(OPCOES_ESCOLARIDADE)][0:2],
    [Botao(rotulo, f"escolaridade:{i}") for i, (rotulo, _) in enumerate(OPCOES_ESCOLARIDADE)][2:4],
]

_BOTOES_ALCANCE: list[list[Botao]] = [
    [Botao(rotulo, f"alcance:{i}") for i, (rotulo, _) in enumerate(OPCOES_ALCANCE)][0:2],
    [Botao(rotulo, f"alcance:{i}") for i, (rotulo, _) in enumerate(OPCOES_ALCANCE)][2:4],
]

# Opção segura (nao-destrutiva) primeiro, e rotulos que restatam a
# consequencia real -- nunca "Sim"/"Nao" vago -- seguindo a diretriz de
# UX pra acoes destrutivas: friccao e clareza proporcionais ao risco.
_BOTOES_REINICIO: list[list[Botao]] = [
    [Botao("Manter meus dados", CALLBACK_REINICIO_CANCELAR)],
    [Botao("Apagar tudo e recomeçar", CALLBACK_REINICIO_CONFIRMAR)],
]

_BOTOES_INICIO: list[list[Botao]] = [
    [Botao("Buscar um curso", CALLBACK_INICIO_BUSCAR)],
    [Botao("Tenho uma dúvida", CALLBACK_INICIO_DUVIDA)],
]

_MENSAGEM_MENU_INICIAL = (
    "Oi! Eu sou o Decifra 😊 Ajudo você a encontrar cursos gratuitos em "
    "institutos federais e a entender editais do IFSC. Como posso te ajudar agora?"
)

_MENSAGEM_MENU_DUVIDA = "Sobre o que você quer saber?"

_MENSAGEM_CONVITE_DUVIDA = (
    "Pode perguntar! Sobre prazo, documento, requisito, o que for -- é só "
    "mandar. E se quiser uma recomendação de curso mais pra frente, também é só pedir."
)

_BOTOES_DUVIDA: list[list[Botao]] = [
    [Botao("Fazer uma pergunta", CALLBACK_DUVIDA_PERGUNTA_LIVRE)],
    [Botao("Guia de cursos", CALLBACK_DUVIDA_GUIA_CURSOS)],
    [Botao("Dúvidas sobre prazos e formas de ingresso", CALLBACK_DUVIDA_PRAZOS)],
]

_MENSAGEM_GUIA_CURSOS = (
    "Aqui está o link com a guia de cursos disponíveis no IFSC: "
    "https://www.ifsc.edu.br/cursos"
)

_BOTOES_POS_EDITAL: list[list[Botao]] = [
    [Botao("Ver outro edital", CALLBACK_EDITAL_VER_OUTRO)],
    [Botao("Encerrar", CALLBACK_EDITAL_ENCERRAR)],
]

_MENSAGEM_ENCERRAMENTO_DUVIDA = "Tudo bem! Se precisar de mais alguma coisa, é só chamar."


def _botoes_lista_editais(editais: list[dict]) -> list[list[Botao]]:
    """Um botão por edital aberto, rótulo = nome real (nunca número).
    callback_data usa o índice na lista carregada agora."""
    return [[Botao(edital["nome"], f"edital:{i}")] for i, edital in enumerate(editais)]


def _mensagem_lista_editais() -> tuple[str, list[list[Botao]]]:
    editais = carregar_editais_abertos()
    if not editais:
        return "No momento não há nenhum edital aberto cadastrado.", []
    return "Qual edital você quer consultar?", _botoes_lista_editais(editais)


def _formatar_detalhe_edital(edital: dict) -> str:
    return (
        f"📅 Prazo de inscrição: {edital['prazo_inicio']} a {edital['prazo_fim']}\n"
        f"🔗 Link para inscrição: {edital['link_inscricao']}\n"
        f"📋 Forma de ingresso: {edital['forma_ingresso']}\n"
        f"📄 Edital completo: {edital['link_pdf']}\n\n"
        "Quer ver outro edital ou posso ajudar em algo mais?"
    )


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
    faltantes = perfil.campos_faltantes()
    contexto = {
        "perfil_atual": perfil.model_dump(),
        "campos_faltantes": faltantes,
    }
    if faltantes and faltantes[0] == "nivel":
        compativeis = niveis_compativeis(perfil.escolaridade)
        contexto["niveis_disponiveis"] = [
            rotulo for rotulo, valor in OPCOES_NIVEL if valor in compativeis
        ]
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


def _responder_via_rag(user_id: str, texto: str) -> str:
    """
    Chama o RAG (`retrieval.generate.answer`) e formata a resposta com
    as fontes citadas. Isolado numa funcao propria pra poder ser
    chamado tanto no fim do fluxo normal (perfil completo/conversa
    livre, pergunta nao-informal) quanto direto na bifurcacao inicial
    (`responder()`) quando a primeira mensagem ja e uma pergunta real
    -- sem essa funcao, os dois pontos de chamada re-avaliariam
    `precisa_busca`/`quer_nova_recomendacao` a toa.
    """
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


def _com_botoes_de_campo_fechado(pergunta: str, perfil: Perfil) -> str | Resposta:
    """
    Envolve a pergunta de coleta com o teclado de botões certo quando o
    próximo campo faltante é de conjunto fechado (hoje: "escolaridade",
    "alcance" e "nivel") -- único ponto que decide se anexa botões,
    usado nos lugares que chamam `_gerar_pergunta_coleta` (coleta
    normal, pós-"buscar outra área"). Campos abertos por design (cidade,
    interesse) nunca ganham botão aqui -- continuam texto livre, como
    sempre foram.
    """
    faltantes = perfil.campos_faltantes()
    if not faltantes:
        return pergunta
    if faltantes[0] == "escolaridade":
        return Resposta(pergunta, botoes=_BOTOES_ESCOLARIDADE)
    if faltantes[0] == "alcance":
        return Resposta(pergunta, botoes=_BOTOES_ALCANCE)
    if faltantes[0] == "nivel":
        return Resposta(pergunta, botoes=_botoes_nivel(perfil.escolaridade))
    return pergunta


def responder(
    user_id: str,
    texto: str,
    sessao: dict,
    nivel_escolhido: str | None = None,
    escolaridade_escolhida: str | None = None,
    alcance_escolhido: str | None = None,
    edital_indice_escolhido: int | None = None,
) -> str:
    """
    Recebe o id do usuário, o texto que ele mandou, e a sessão atual.

    `nivel_escolhido`/`escolaridade_escolhida`/`alcance_escolhido` são
    preenchidos só quando a origem da chamada foi um toque num botão
    inline correspondente (`channels/telegram.py`): pulam a chamada ao
    extrator (LLM) e definem o campo direto com o valor do botão -- mais
    barato e determinístico que tratar o rótulo do botão como se a
    pessoa tivesse digitado. Por construção, esses botões só existem
    durante a fase "coletando", nunca podem ser um pedido de reinício,
    então também pulam os blocos de reinício abaixo
    (`classificar_pedido_reinicio` custaria uma chamada paga à toa nesse
    caso).

    `edital_indice_escolhido` é preenchido só quando a origem foi um
    toque num botão de edital específico (dentro do sub-menu "Dúvidas
    sobre prazos e formas de ingresso") -- resolvido logo no início,
    antes de qualquer outro roteamento, e não depende de `fase_dialogo`
    estar em nenhum valor específico (mesma filosofia dos outros
    parâmetros de botão).

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

    Em seguida, se `fase_dialogo == "inicio"` (sessão nova, nunca
    processou mensagem nenhuma), resolve a bifurcação inicial: pergunta
    direta sobre edital pula a coleta e vai pro RAG direto
    (`fase_dialogo` vira "conversa_livre" -- perfil fica vazio, toda
    mensagem futura é tratada como candidata a RAG); pedido de
    recomendação (texto livre ou botão) entra na coleta normal
    (`fase_dialogo` vira "coletando"); "tenho uma dúvida" mostra o
    sub-menu (guia de cursos / prazos e formas de ingresso,
    `fase_dialogo` vira "menu_duvida"); qualquer outra coisa (saudação,
    texto ambíguo) mostra o menu inicial com botões. O sub-menu de
    dúvida ("menu_duvida"/"selecionando_edital") tem sua própria lógica
    de navegação: guia de cursos responde direto e volta pra
    "conversa_livre"; prazos mostra a lista de editais abertos
    ("selecionando_edital"); escolher um edital (via
    `edital_indice_escolhido`) mostra o detalhe formatado com botões de
    "ver outro"/"encerrar". Alguém em "conversa_livre" que depois pede
    uma recomendação (`quer_nova_recomendacao`) migra pra coleta na
    hora, mesmo com o perfil ainda vazio -- nunca fica sem essa saída.

    Se o perfil ainda não estiver completo (e não estiver em
    "conversa_livre"), extrai o que der da mensagem (com o histórico
    recente como contexto pra resolver referências tipo "e advogado?"),
    atualiza a sessão (in-place -- quem chama esta função é responsável
    por persistir a sessão de volta no Redis) e devolve a próxima
    pergunta de coleta. Quando o perfil acaba de ficar completo neste
    turno, devolve a recomendação do motor estruturado
    (`recommend/opportunities.py`). Com o perfil já completo de antes
    (ou em "conversa_livre"), uma nova recomendação só é gerada se a
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

    # Gatilho explícito de reinício total ("recomeçar" e variações
    # fortes) tem prioridade sobre QUALQUER roteamento normal, em
    # qualquer fase -- coleta (mesmo no meio de uma pergunta específica
    # de campo), RAG, ou perfil já completo. Deterministico (regex, sem
    # LLM): ver `dialogue.reset.eh_gatilho_explicito_de_reinicio_total`.
    # Não interrompe uma confirmação já em andamento -- essa segue pro
    # bloco de baixo, que trata a resposta como confirmação/negação.
    if (
        texto
        and sessao.get("fase_dialogo") != "confirmando_reinicio"
        and eh_gatilho_explicito_de_reinicio_total(texto)
    ):
        sessao["fase_dialogo_anterior"] = sessao.get("fase_dialogo")
        sessao["fase_dialogo"] = "confirmando_reinicio"
        try:
            pergunta = _gerar_confirmacao_reinicio(texto)
        except Exception as exc:
            _logar_falha("gerar confirmacao de reinicio", user_id, exc)
            return Resposta(_MENSAGEM_FALLBACK_CONFIRMACAO_REINICIO, botoes=_BOTOES_REINICIO)
        return Resposta(pergunta, botoes=_BOTOES_REINICIO)

    # Seleção de edital específico no sub-menu de dúvidas: resolvida
    # logo no início, como os demais parâmetros de botão (nivel_
    # escolhido/escolaridade_escolhida/alcance_escolhido) -- não
    # depende de fase_dialogo estar em nenhum valor específico.
    if edital_indice_escolhido is not None:
        edital = buscar_edital_por_indice(edital_indice_escolhido)
        if edital is None:
            return _MENSAGEM_FALLBACK
        return Resposta(_formatar_detalhe_edital(edital), botoes=_BOTOES_POS_EDITAL)

    if nivel_escolhido is None and escolaridade_escolhida is None and alcance_escolhido is None:
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

        # Sub-menu "Tenho uma dúvida": guia de cursos responde direto
        # (sem sub-menu); prazos mostra a lista de editais abertos.
        if sessao.get("fase_dialogo") == "menu_duvida":
            if texto == TEXTO_SINTETICO_PERGUNTA_LIVRE:
                sessao["fase_dialogo"] = "conversa_livre"
                return _MENSAGEM_CONVITE_DUVIDA
            if texto == TEXTO_SINTETICO_GUIA_CURSOS:
                sessao["fase_dialogo"] = "conversa_livre"
                return _MENSAGEM_GUIA_CURSOS
            if texto == TEXTO_SINTETICO_DUVIDA_PRAZOS:
                sessao["fase_dialogo"] = "selecionando_edital"
                mensagem, botoes = _mensagem_lista_editais()
                return Resposta(mensagem, botoes=botoes) if botoes else mensagem

        # Navegação pós-detalhe de edital: ver outro (mostra a lista de
        # novo) ou encerrar (volta pra conversa livre).
        if sessao.get("fase_dialogo") == "selecionando_edital":
            if texto == TEXTO_SINTETICO_VER_OUTRO_EDITAL:
                mensagem, botoes = _mensagem_lista_editais()
                return Resposta(mensagem, botoes=botoes) if botoes else mensagem
            if texto == TEXTO_SINTETICO_ENCERRAR_DUVIDA:
                sessao["fase_dialogo"] = "conversa_livre"
                return _MENSAGEM_ENCERRAMENTO_DUVIDA

        # So considera pedido de reinicio com o perfil ja completo.
        # "buscar outra area" (trocar de area preservando cidade/
        # escolaridade/alcance) e "comecar de novo" (limpar tudo) so
        # fazem sentido de verdade depois que ha algo definido pra
        # trocar/descartar -- e, na pratica, o classificador (que so
        # ve o texto solto, sem saber que pergunta o bot acabou de
        # fazer) e pouco confiavel durante a coleta: confirmado ao
        # vivo que "quero tecnologia" (primeira resposta de interesse)
        # e "topo ir pra uma cidade proxima" (resposta de alcance)
        # foram classificados incorretamente como buscar_outra_area,
        # descartando a mensagem e travando a coleta em loop. Corrigir
        # um campo errado durante a coleta ja e resolvido pelo proprio
        # extrator (ele sobrescreve quando a pessoa da um valor novo),
        # entao restringir o classificador ao perfil completo elimina
        # os falsos positivos sem perder um caso de uso real.
        perfil_para_checar_reinicio = Perfil(**(sessao.get("perfil") or {}))
        if determinar_fase(perfil_para_checar_reinicio) == "completo":
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
                return _com_botoes_de_campo_fechado(pergunta, perfil_pos_reset)
            elif pedido == "comecar_de_novo":
                sessao["fase_dialogo_anterior"] = sessao.get("fase_dialogo")
                sessao["fase_dialogo"] = "confirmando_reinicio"
                try:
                    pergunta = _gerar_confirmacao_reinicio(texto)
                except Exception as exc:
                    _logar_falha("gerar confirmacao de reinicio", user_id, exc)
                    return Resposta(_MENSAGEM_FALLBACK_CONFIRMACAO_REINICIO, botoes=_BOTOES_REINICIO)
                return Resposta(pergunta, botoes=_BOTOES_REINICIO)

        # Bifurcacao inicial: sessao nova (fase_dialogo == "inicio",
        # default de carregar_sessao) nunca passou por essa decisao
        # ainda. Sem isso, toda conversa nova entra direto em coleta
        # de perfil, mesmo quando a primeira mensagem ja e uma
        # pergunta direta sobre edital -- obrigando a pessoa a
        # responder cidade/escolaridade/etc antes de conseguir
        # perguntar o que queria. Os textos sinteticos dos botoes
        # ("quero buscar um curso"/"tenho uma duvida") usam == direto,
        # deterministico, igual nivel_escolhido/TEXTO_SINTETICO_* do
        # reinicio -- nunca depende de um classificador pago adivinhar
        # certo pra algo que a propria UI ja sabe. Texto livre organico
        # continua passando pelos classificadores de verdade
        # (quer_nova_recomendacao, precisa_busca -- ja existentes, sem
        # nenhum classificador novo).
        if sessao.get("fase_dialogo") == "inicio":
            if texto == TEXTO_SINTETICO_TENHO_DUVIDA:
                sessao["fase_dialogo"] = "menu_duvida"
                return Resposta(_MENSAGEM_MENU_DUVIDA, botoes=_BOTOES_DUVIDA)

            quer_recomendacao = texto == TEXTO_SINTETICO_BUSCAR_CURSO or quer_nova_recomendacao(texto)
            if quer_recomendacao:
                sessao["fase_dialogo"] = "coletando"
                # cai no fluxo normal abaixo, que ja inicia a coleta
            elif precisa_busca(texto):
                # Pergunta direta de cara -- ja sabemos que nao e
                # pedido de recomendacao (quer_recomendacao acima) nem
                # duvida generica, entao responde via RAG direto aqui,
                # sem cair no fluxo normal (que re-chamaria
                # quer_nova_recomendacao/precisa_busca a toa).
                sessao["fase_dialogo"] = "conversa_livre"
                return _responder_via_rag(user_id, texto)
            else:
                return Resposta(_MENSAGEM_MENU_INICIAL, botoes=_BOTOES_INICIO)

    perfil_atual = sessao.get("perfil") or {}
    perfil = Perfil(**perfil_atual)

    if sessao.get("fase_dialogo") != "conversa_livre" and determinar_fase(perfil) != "completo":
        historico = sessao.get("historico") or []
        # Botão obsoleto (ex.: a pessoa reiniciou e a coleta recomeçou
        # do zero, mas ainda tinha um teclado antigo na tela): a fase
        # pode continuar "coletando" depois do reinício, então só isso
        # não basta pra confirmar que o botão ainda vale -- confere
        # também se o campo do botão ainda é de fato o próximo
        # faltante. Se não for, cai no mesmo fallback dos outros casos
        # de botão obsoleto: trata o rótulo como texto livre normal.
        faltantes_atuais = perfil.campos_faltantes()
        proximo_campo_esperado = faltantes_atuais[0] if faltantes_atuais else None
        if nivel_escolhido and proximo_campo_esperado == "nivel":
            perfil = Perfil(**{**perfil_atual, "nivel": nivel_escolhido})
        elif escolaridade_escolhida and proximo_campo_esperado == "escolaridade":
            mesclado = aplicar_coerencia_nivel({**perfil_atual, "escolaridade": escolaridade_escolhida})
            perfil = Perfil(**mesclado)
        elif alcance_escolhido and proximo_campo_esperado == "alcance":
            perfil = Perfil(**{**perfil_atual, "alcance": alcance_escolhido})
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
            return _com_botoes_de_campo_fechado(pergunta, perfil)

        try:
            return gerar_recomendacao(perfil)
        except Exception as exc:
            _logar_falha("gerar recomendação", user_id, exc)
            return _MENSAGEM_FALLBACK

    # Bifurcação motor estruturado (recommend/) vs RAG (retrieval/): daqui
    # pra baixo é o ponto de roteamento semântico que um futuro Agente
    # Supervisor multi-agente assumiria -- `quer_nova_recomendacao` decide
    # se cai no motor estruturado (sem LLM decidindo prazo/data, só
    # redigindo o resultado pronto), senão `precisa_busca` decide entre
    # RAG (`_responder_via_rag`) e conversa livre.
    if quer_nova_recomendacao(texto):
        try:
            historico = sessao.get("historico") or []
            perfil = extrair_perfil(texto, perfil.model_dump(), historico=historico)
            sessao["perfil"] = perfil.model_dump()
            # Saida de escape do "conversa_livre": a pessoa que tinha
            # pulado a coleta (ou ainda estava com perfil incompleto)
            # pode pedir uma recomendacao a qualquer momento -- migra
            # pra coleta normal em vez de tentar recomendar com dado
            # faltando.
            if determinar_fase(perfil) != "completo":
                sessao["fase_dialogo"] = "coletando"
                pergunta = _gerar_pergunta_coleta(perfil)
                return _com_botoes_de_campo_fechado(pergunta, perfil)
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

    return _responder_via_rag(user_id, texto)