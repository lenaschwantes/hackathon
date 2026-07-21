"""
Testes do adaptador de Telegram (`channels/telegram.py`) -- foco no
handler de botões inline (`_ao_receber_botao`), que hoje não tinha
nenhuma cobertura dedicada. Não toca Anthropic, Weaviate nem Redis de
verdade: usa `MagicMock`/`AsyncMock`, mesmo padrão de
`tests/test_seguranca.py`.
"""

import asyncio

from unittest.mock import AsyncMock, MagicMock

import pytest

from channels import engine
from channels import session as session_module
from channels.telegram import TelegramAdapter


class _FakeRedis:
    """Fake mínimo do cliente Redis -- só o suficiente pra
    carregar_sessao/salvar_sessao/permitido funcionarem."""

    def __init__(self):
        self._store: dict[str, str] = {}
        self._contadores: dict[str, int] = {}

    async def get(self, chave):
        return self._store.get(chave)

    async def set(self, chave, valor, ex=None):
        self._store[chave] = valor

    async def incr(self, chave):
        self._contadores[chave] = self._contadores.get(chave, 0) + 1
        return self._contadores[chave]

    async def expire(self, chave, segundos):
        pass


@pytest.fixture
def fake_redis(monkeypatch):
    fake = _FakeRedis()
    monkeypatch.setattr(session_module, "_redis", fake)
    return fake


def _fake_callback_update(user_id: int, data: str):
    """Duck-type mínimo do `telegram.Update` pra um callback_query."""
    update = MagicMock()
    update.callback_query.data = data
    update.callback_query.from_user.id = user_id
    update.callback_query.answer = AsyncMock()
    update.callback_query.edit_message_reply_markup = AsyncMock()
    update.callback_query.message.chat_id = user_id
    return update


def _fake_context():
    context = MagicMock()
    context.bot.send_message = AsyncMock()
    return context


def _sessao(perfil: dict, fase: str = "coletando") -> dict:
    return {"perfil": perfil, "fase_dialogo": fase, "historico": []}


_PERFIL_SO_FALTA_NIVEL = {
    "cidade": "Blumenau",
    "escolaridade": "ensino medio completo",
    "interesse": "tecnologia",
    "nivel": None,
    "modalidade": None,
    "alcance": "regional",
}

_PERFIL_COMPLETO = {
    "cidade": "Blumenau",
    "escolaridade": "ensino medio completo",
    "interesse": "tecnologia",
    "nivel": "tecnico integrado",
    "modalidade": None,
    "alcance": "regional",
}


async def _salvar_sessao_inicial(user_id: str, sessao: dict) -> None:
    from channels.session import salvar_sessao

    await salvar_sessao(user_id, sessao)


class TestBotaoDeNivel:
    def test_botao_de_nivel_chama_responder_com_nivel_escolhido(self, fake_redis, monkeypatch):
        monkeypatch.setenv("TELEGRAM_BOT_TOKEN", "dummy:token-para-teste")
        capturado = {}

        def fake_responder(user_id, texto, sessao, nivel_escolhido=None):
            capturado["texto"] = texto
            capturado["nivel_escolhido"] = nivel_escolhido
            return "Perfeito, já registrei!"

        adapter = TelegramAdapter(responder=fake_responder)

        async def cenario():
            await _salvar_sessao_inicial("55", _sessao(dict(_PERFIL_SO_FALTA_NIVEL)))
            update = _fake_callback_update(user_id=55, data="nivel:0")
            await adapter._ao_receber_botao(update, _fake_context())

        asyncio.run(cenario())

        assert capturado["nivel_escolhido"] == "tecnico integrado"
        assert capturado["texto"] == "Técnico integrado"

    def test_botao_de_nivel_stale_nao_forca_nivel_quando_fase_mudou(self, fake_redis, monkeypatch):
        monkeypatch.setenv("TELEGRAM_BOT_TOKEN", "dummy:token-para-teste")
        capturado = {}

        def fake_responder(user_id, texto, sessao, nivel_escolhido=None):
            capturado["nivel_escolhido"] = nivel_escolhido
            return "ok"

        adapter = TelegramAdapter(responder=fake_responder)

        async def cenario():
            # Sessão já completou (ex.: a pessoa reiniciou) antes do
            # toque no botão de nível chegar -- botão obsoleto.
            await _salvar_sessao_inicial("56", _sessao(dict(_PERFIL_COMPLETO), fase="completo"))
            update = _fake_callback_update(user_id=56, data="nivel:1")
            await adapter._ao_receber_botao(update, _fake_context())

        asyncio.run(cenario())

        assert capturado["nivel_escolhido"] is None


class TestBotaoDeReinicio:
    def test_botao_de_confirmar_reinicio_usa_texto_sintetico_confirmar(self, fake_redis, monkeypatch):
        monkeypatch.setenv("TELEGRAM_BOT_TOKEN", "dummy:token-para-teste")
        capturado = {}

        def fake_responder(user_id, texto, sessao, nivel_escolhido=None):
            capturado["texto"] = texto
            return "Prontinho, apaguei tudo!"

        adapter = TelegramAdapter(responder=fake_responder)

        async def cenario():
            await _salvar_sessao_inicial("57", _sessao(dict(_PERFIL_COMPLETO), fase="confirmando_reinicio"))
            update = _fake_callback_update(user_id=57, data=engine.CALLBACK_REINICIO_CONFIRMAR)
            await adapter._ao_receber_botao(update, _fake_context())

        asyncio.run(cenario())

        assert capturado["texto"] == "confirmo"

    def test_botao_de_cancelar_reinicio_usa_texto_sintetico_cancelar(self, fake_redis, monkeypatch):
        monkeypatch.setenv("TELEGRAM_BOT_TOKEN", "dummy:token-para-teste")
        capturado = {}

        def fake_responder(user_id, texto, sessao, nivel_escolhido=None):
            capturado["texto"] = texto
            return "Sem problema, mantive seus dados como estavam."

        adapter = TelegramAdapter(responder=fake_responder)

        async def cenario():
            await _salvar_sessao_inicial("58", _sessao(dict(_PERFIL_COMPLETO), fase="confirmando_reinicio"))
            update = _fake_callback_update(user_id=58, data=engine.CALLBACK_REINICIO_CANCELAR)
            await adapter._ao_receber_botao(update, _fake_context())

        asyncio.run(cenario())

        assert capturado["texto"] == "manter meus dados"


class TestComportamentoDoTeclado:
    def test_botao_apertado_remove_teclado_da_mensagem_original(self, fake_redis, monkeypatch):
        monkeypatch.setenv("TELEGRAM_BOT_TOKEN", "dummy:token-para-teste")
        adapter = TelegramAdapter(responder=lambda user_id, texto, sessao, nivel_escolhido=None: "ok")

        async def cenario():
            await _salvar_sessao_inicial("59", _sessao(dict(_PERFIL_SO_FALTA_NIVEL)))
            update = _fake_callback_update(user_id=59, data="nivel:2")
            await adapter._ao_receber_botao(update, _fake_context())
            return update

        update = asyncio.run(cenario())

        update.callback_query.edit_message_reply_markup.assert_awaited_once_with(reply_markup=None)

    def test_resposta_com_botoes_monta_inline_keyboard_correto(self, fake_redis, monkeypatch):
        monkeypatch.setenv("TELEGRAM_BOT_TOKEN", "dummy:token-para-teste")
        botoes = [[engine.Botao("A", "x"), engine.Botao("B", "y")]]

        def fake_responder(user_id, texto, sessao, nivel_escolhido=None):
            return engine.Resposta("pergunta", botoes=botoes)

        adapter = TelegramAdapter(responder=fake_responder)

        async def cenario():
            await _salvar_sessao_inicial("60", _sessao(dict(_PERFIL_SO_FALTA_NIVEL)))
            update = _fake_callback_update(user_id=60, data="nivel:3")
            context = _fake_context()
            await adapter._ao_receber_botao(update, context)
            return context

        context = asyncio.run(cenario())

        _, kwargs = context.bot.send_message.call_args
        teclado = kwargs["reply_markup"]
        assert [b.callback_data for b in teclado.inline_keyboard[0]] == ["x", "y"]
        assert [b.text for b in teclado.inline_keyboard[0]] == ["A", "B"]
