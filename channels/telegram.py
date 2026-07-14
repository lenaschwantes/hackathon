"""
Adaptador do canal Telegram. Só este arquivo pode importar
qualquer coisa relacionada a Telegram — nenhum outro arquivo do
projeto deve fazer isso.
"""

import asyncio
import os

from telegram import Update
from telegram.ext import Application, ContextTypes, MessageHandler, filters

from channels.base import ChannelAdapter
from channels.engine import responder as fake_responder
from channels.rate_limit import MENSAGEM_LIMITE_EXCEDIDO, permitido
from channels.session import carregar_sessao, salvar_sessao

# Teto de tamanho de mensagem: sem isso, uma mensagem de qualquer
# tamanho chega inteira nos motores (Groq/Voyage, cobrados por token),
# expondo o pipeline a abuso via mensagens gigantes.
_MAX_CARACTERES_MENSAGEM = 4000


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
        self._app.add_handler(
            MessageHandler(filters.TEXT & ~filters.COMMAND, self._ao_receber)
        )

    async def _ao_receber(self, update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
        """
        Chamado a cada mensagem de texto recebida. Faz o ciclo
        completo: checa rate limit -> carrega sessão -> chama o motor
        -> salva sessão -> responde. Nenhuma lógica de negócio mora
        aqui, só a orquestração entre sessão e motor.
        """
        user_id = str(update.effective_user.id)
        texto = update.message.text[:_MAX_CARACTERES_MENSAGEM]

        if not await permitido(user_id):
            await update.message.reply_text(MENSAGEM_LIMITE_EXCEDIDO)
            return

        sessao = await carregar_sessao(user_id)
        resposta = self._responder(user_id, texto, sessao)

        sessao["historico"].append({"de": "usuario", "texto": texto})
        sessao["historico"].append({"de": "bot", "texto": resposta})
        await salvar_sessao(user_id, sessao)

        await update.message.reply_text(resposta)

    async def enviar(self, user_id: str, texto: str) -> None:
        await self._app.bot.send_message(chat_id=int(user_id), text=texto)

    async def iniciar(self) -> None:
        await self._app.initialize()
        await self._app.start()
        await self._app.updater.start_polling()
        # mantém o bot ligado indefinidamente, escutando mensagens
        while True:
            await asyncio.sleep(3600)