"""
Adaptador do canal Telegram. Só este arquivo pode importar
qualquer coisa relacionada a Telegram — nenhum outro arquivo do
projeto deve fazer isso.
"""

import asyncio
import logging
import os

from telegram import InlineKeyboardButton, InlineKeyboardMarkup, Update
from telegram.ext import (
    Application,
    CallbackQueryHandler,
    CommandHandler,
    ContextTypes,
    MessageHandler,
    filters,
)

from channels.base import ChannelAdapter
from channels.engine import _BOTOES_INICIO, _MAX_CARACTERES_MENSAGEM, _MENSAGEM_MENU_INICIAL, Botao
from channels.engine import responder as fake_responder
from channels.session import carregar_sessao, salvar_sessao
from dialogue.editais_catalogo import buscar_edital_por_indice
from dialogue.onboarding import (
    CALLBACK_INICIO_BUSCAR,
    CALLBACK_INICIO_DUVIDA,
    CALLBACK_DUVIDA_GUIA_CURSOS,
    CALLBACK_DUVIDA_PRAZOS,
    CALLBACK_EDITAL_VER_OUTRO,
    CALLBACK_EDITAL_ENCERRAR,
    TEXTO_SINTETICO_BUSCAR_CURSO,
    TEXTO_SINTETICO_TENHO_DUVIDA,
    TEXTO_SINTETICO_GUIA_CURSOS,
    TEXTO_SINTETICO_DUVIDA_PRAZOS,
    TEXTO_SINTETICO_VER_OUTRO_EDITAL,
    TEXTO_SINTETICO_ENCERRAR_DUVIDA,
)
from dialogue.profile import OPCOES_ALCANCE, OPCOES_ESCOLARIDADE, OPCOES_NIVEL
from dialogue.reset import (
    CALLBACK_REINICIO_CANCELAR,
    CALLBACK_REINICIO_CONFIRMAR,
    TEXTO_SINTETICO_CANCELAR,
    TEXTO_SINTETICO_CONFIRMAR,
)
from infra.rate_limit import MENSAGEM_LIMITE_EXCEDIDO, eh_duplicada, permitido

logger = logging.getLogger(__name__)


def _montar_teclado(botoes: list[list[Botao]] | None) -> InlineKeyboardMarkup | None:
    """Converte a lista de `Botao` (agnóstica de Telegram) num `InlineKeyboardMarkup`."""
    if not botoes:
        return None
    return InlineKeyboardMarkup(
        [[InlineKeyboardButton(botao.rotulo, callback_data=botao.callback_data) for botao in linha] for linha in botoes]
    )


class TelegramAdapter(ChannelAdapter):
    def __init__(self, responder=None):
        """
        `responder` é a função injetada que gera a resposta.
        Durante o desenvolvimento, usa a do fake_engine por padrão.
        Quando o motor de verdade estiver pronto, basta passar ele
        aqui em vez do fake — este arquivo não muda.
        """
        token = os.environ["TELEGRAM_BOT_TOKEN"]
        self._responder = responder or fake_responder
        self._app = Application.builder().token(token).build()
        self._app.add_handler(CommandHandler("start", self._ao_receber_start))
        self._app.add_handler(CommandHandler("recomecar", self._ao_receber_comando_recomecar))
        self._app.add_handler(
            MessageHandler(filters.TEXT & ~filters.COMMAND, self._ao_receber)
        )
        self._app.add_handler(CallbackQueryHandler(self._ao_receber_botao))

    async def _ao_receber_start(self, update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
        """
        `/start` e o gesto padrao do Telegram pra comecar (ou
        recomecar) a interagir com um bot -- o `MessageHandler` normal
        nunca ve isso, `filters.TEXT & ~filters.COMMAND` exclui
        comandos de proposito. Mostra o menu de abertura direto, sem
        precisar de nenhum classificador (a pessoa ja deixou a intencao
        clara so de mandar o comando). Nao apaga perfil nem historico
        -- isso e papel do reinicio "apagar tudo" (botao/texto
        dedicado); `/start` so marca `fase_dialogo` de volta pra
        "inicio", pra a proxima mensagem ou toque de botao resolver a
        bifurcacao normalmente (ver `channels/engine.py::responder`).
        """
        user_id = str(update.effective_user.id)

        if not await permitido(user_id):
            await update.message.reply_text(MENSAGEM_LIMITE_EXCEDIDO)
            return

        sessao = await carregar_sessao(user_id)
        sessao["fase_dialogo"] = "inicio"
        await salvar_sessao(user_id, sessao)

        await update.message.reply_text(_MENSAGEM_MENU_INICIAL, reply_markup=_montar_teclado(_BOTOES_INICIO))

    async def _processar_mensagem_de_texto(self, user_id: str, texto: str, update: Update) -> None:
        """
        Corpo comum entre uma mensagem de texto normal e o comando
        `/recomecar` (que só fixa o texto e reaproveita o resto do
        ciclo): checa dedup/rate limit -> carrega sessão -> chama o
        motor -> salva sessão -> responde. Nenhuma lógica de negócio
        mora aqui, só a orquestração entre sessão e motor.
        """
        if await eh_duplicada(user_id, texto):
            await update.message.reply_text(
                "Já tô cuidando disso pra você! Me dá um instante."
            )
            return

        if not await permitido(user_id):
            await update.message.reply_text(MENSAGEM_LIMITE_EXCEDIDO)
            return

        sessao = await carregar_sessao(user_id)
        resposta = self._responder(user_id, texto, sessao)

        sessao["historico"].append({"de": "usuario", "texto": texto})
        sessao["historico"].append({"de": "bot", "texto": str(resposta)})
        await salvar_sessao(user_id, sessao)

        botoes = getattr(resposta, "botoes", None)
        await update.message.reply_text(str(resposta), reply_markup=_montar_teclado(botoes))

    async def _ao_receber(self, update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
        """Chamado a cada mensagem de texto recebida."""
        user_id = str(update.effective_user.id)
        texto = update.message.text[:_MAX_CARACTERES_MENSAGEM]
        await self._processar_mensagem_de_texto(user_id, texto, update)

    async def _ao_receber_comando_recomecar(self, update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
        """
        `/recomecar` -- mesmo gatilho de "recomeçar" digitado (ver
        `dialogue.reset.eh_gatilho_explicito_de_reinicio_total`), só
        que via comando explícito do Telegram (que o `MessageHandler`
        normal nunca vê, `filters.TEXT & ~filters.COMMAND` exclui
        comandos de propósito) em vez de mensagem de texto solta.
        """
        user_id = str(update.effective_user.id)
        await self._processar_mensagem_de_texto(user_id, "recomeçar", update)

    async def _ao_receber_botao(self, update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
        """
        Chamado a cada toque num botão inline (seleção de nível ou
        confirmação de reinício). Reaproveita o mesmo `self._responder`
        e a mesma sessão do canal de texto -- um toque de botão produz
        exatamente o mesmo estado de sessão que digitar a resposta
        equivalente produziria (ver `channels/engine.py::responder`).
        """
        query = update.callback_query
        await query.answer()  # limpa o spinner de carregamento do cliente Telegram

        user_id = str(query.from_user.id)
        data = query.data

        # Mesma proteção de dedup do canal de texto (`_ao_receber`) --
        # sem isso, um duplo toque rápido no mesmo botão (ex.: rede
        # lenta, a pessoa toca de novo achando que não registrou)
        # processa a escolha duas vezes. Chave pelo `callback_data`
        # (ex.: "nivel:2"), não pelo rótulo -- é o identificador real
        # do que foi clicado.
        if data is not None and await eh_duplicada(user_id, data):
            await context.bot.send_message(
                chat_id=query.message.chat_id, text="Já tô cuidando disso pra você! Me dá um instante."
            )
            return

        if not await permitido(user_id):
            await context.bot.send_message(chat_id=query.message.chat_id, text=MENSAGEM_LIMITE_EXCEDIDO)
            return

        sessao = await carregar_sessao(user_id)

        if data is not None and data.startswith("nivel:"):
            indice = int(data.split(":", 1)[1])
            rotulo, valor = OPCOES_NIVEL[indice]
            if sessao.get("fase_dialogo") == "coletando":
                texto_usuario = rotulo
                resposta = self._responder(user_id, rotulo, sessao, valor)
            else:
                # Botão obsoleto: a sessão mudou de fase desde que ele
                # apareceu (ex.: a pessoa reiniciou nesse meio tempo).
                # Não força mais um nível numa fase a que ele não
                # pertence -- cai no fluxo normal com o texto do botão.
                texto_usuario = rotulo
                resposta = self._responder(user_id, rotulo, sessao)
        elif data is not None and data.startswith("escolaridade:"):
            indice = int(data.split(":", 1)[1])
            rotulo, valor = OPCOES_ESCOLARIDADE[indice]
            if sessao.get("fase_dialogo") == "coletando":
                texto_usuario = rotulo
                resposta = self._responder(user_id, rotulo, sessao, escolaridade_escolhida=valor)
            else:
                # Mesmo caso do botão de nível obsoleto acima: sessão
                # mudou de fase desde que o botão apareceu -- cai no
                # fluxo normal com o texto do botão.
                texto_usuario = rotulo
                resposta = self._responder(user_id, rotulo, sessao)
        elif data is not None and data.startswith("alcance:"):
            indice = int(data.split(":", 1)[1])
            rotulo, valor = OPCOES_ALCANCE[indice]
            if sessao.get("fase_dialogo") == "coletando":
                texto_usuario = rotulo
                resposta = self._responder(user_id, rotulo, sessao, alcance_escolhido=valor)
            else:
                # Mesmo caso dos botões obsoletos acima.
                texto_usuario = rotulo
                resposta = self._responder(user_id, rotulo, sessao)
        elif data in (CALLBACK_REINICIO_CONFIRMAR, CALLBACK_REINICIO_CANCELAR):
            texto_usuario = (
                TEXTO_SINTETICO_CONFIRMAR if data == CALLBACK_REINICIO_CONFIRMAR else TEXTO_SINTETICO_CANCELAR
            )
            resposta = self._responder(user_id, texto_usuario, sessao)
        elif data in (CALLBACK_INICIO_BUSCAR, CALLBACK_INICIO_DUVIDA):
            texto_usuario = (
                TEXTO_SINTETICO_BUSCAR_CURSO if data == CALLBACK_INICIO_BUSCAR else TEXTO_SINTETICO_TENHO_DUVIDA
            )
            resposta = self._responder(user_id, texto_usuario, sessao)
        elif data in (CALLBACK_DUVIDA_GUIA_CURSOS, CALLBACK_DUVIDA_PRAZOS):
            texto_usuario = (
                TEXTO_SINTETICO_GUIA_CURSOS if data == CALLBACK_DUVIDA_GUIA_CURSOS else TEXTO_SINTETICO_DUVIDA_PRAZOS
            )
            resposta = self._responder(user_id, texto_usuario, sessao)
        elif data in (CALLBACK_EDITAL_VER_OUTRO, CALLBACK_EDITAL_ENCERRAR):
            texto_usuario = (
                TEXTO_SINTETICO_VER_OUTRO_EDITAL if data == CALLBACK_EDITAL_VER_OUTRO else TEXTO_SINTETICO_ENCERRAR_DUVIDA
            )
            resposta = self._responder(user_id, texto_usuario, sessao)
        elif data is not None and data.startswith("edital:"):
            indice = int(data.split(":", 1)[1])
            edital = buscar_edital_por_indice(indice)
            texto_usuario = edital["nome"] if edital else f"edital #{indice}"
            resposta = self._responder(user_id, texto_usuario, sessao, edital_indice_escolhido=indice)
        else:
            logger.warning("callback_data desconhecido recebido: %r", data)
            return

        sessao["historico"].append({"de": "usuario", "texto": texto_usuario})
        sessao["historico"].append({"de": "bot", "texto": str(resposta)})
        await salvar_sessao(user_id, sessao)

        try:
            await query.edit_message_reply_markup(reply_markup=None)
        except Exception:
            # Mensagem original já editada/apagada -- não deve derrubar
            # a resposta por causa disso, só deixa de remover o teclado.
            pass

        botoes = getattr(resposta, "botoes", None)
        await context.bot.send_message(
            chat_id=query.message.chat_id, text=str(resposta), reply_markup=_montar_teclado(botoes)
        )

    async def enviar(self, user_id: str, texto: str) -> None:
        await self._app.bot.send_message(chat_id=int(user_id), text=texto)

    async def iniciar(self) -> None:
        await self._app.initialize()
        await self._app.start()
        await self._app.updater.start_polling()
        # mantém o bot ligado indefinidamente, escutando mensagens
        while True:
            await asyncio.sleep(3600)