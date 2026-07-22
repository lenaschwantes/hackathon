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

    async def set(self, chave, valor, ex=None, nx=False):
        if nx and chave in self._store:
            return None
        self._store[chave] = valor
        return True

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


def _fake_update_comando(user_id: int):
    """Duck-type mínimo do `telegram.Update` pra um comando (ex.: /start)."""
    update = MagicMock()
    update.effective_user.id = user_id
    update.message.reply_text = AsyncMock()
    return update


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


class TestTodosOsBotoesDeNivelExtraemValorCorreto:
    """Regressão do bug reportado ('Graduação' oferecendo nível de
    ensino médio depois): confirma, pra CADA botão do teclado de
    nível -- não só 'Graduação' -- que o índice do callback_data
    resolve pro par rótulo/valor certo em `OPCOES_NIVEL`, sem
    dessincronia entre a lista usada pro rótulo e a usada pro valor."""

    @pytest.mark.parametrize(
        "indice,rotulo_esperado,valor_esperado",
        [
            (0, "Técnico integrado", "tecnico integrado"),
            (1, "Técnico subsequente", "tecnico subsequente"),
            (2, "Graduação", "superior"),
            (3, "FIC (curso rápido)", "FIC"),
        ],
    )
    def test_cada_botao_de_nivel_extrai_o_par_certo(
        self, fake_redis, monkeypatch, indice, rotulo_esperado, valor_esperado
    ):
        monkeypatch.setenv("TELEGRAM_BOT_TOKEN", "dummy:token-para-teste")
        capturado = {}

        def fake_responder(user_id, texto, sessao, nivel_escolhido=None, escolaridade_escolhida=None):
            capturado["texto"] = texto
            capturado["nivel_escolhido"] = nivel_escolhido
            return "ok"

        adapter = TelegramAdapter(responder=fake_responder)

        async def cenario():
            await _salvar_sessao_inicial("60", _sessao(dict(_PERFIL_SO_FALTA_NIVEL)))
            update = _fake_callback_update(user_id=60, data=f"nivel:{indice}")
            await adapter._ao_receber_botao(update, _fake_context())

        asyncio.run(cenario())

        assert capturado["nivel_escolhido"] == valor_esperado
        assert capturado["texto"] == rotulo_esperado


class TestBotaoDeEscolaridade:
    def test_botao_de_escolaridade_chama_responder_com_escolaridade_escolhida(self, fake_redis, monkeypatch):
        monkeypatch.setenv("TELEGRAM_BOT_TOKEN", "dummy:token-para-teste")
        capturado = {}

        def fake_responder(user_id, texto, sessao, nivel_escolhido=None, escolaridade_escolhida=None):
            capturado["texto"] = texto
            capturado["escolaridade_escolhida"] = escolaridade_escolhida
            return "Perfeito, já registrei!"

        adapter = TelegramAdapter(responder=fake_responder)

        async def cenario():
            perfil_faltando_escolaridade = {**_PERFIL_SO_FALTA_NIVEL, "escolaridade": None, "nivel": None}
            await _salvar_sessao_inicial("61", _sessao(perfil_faltando_escolaridade))
            update = _fake_callback_update(user_id=61, data="escolaridade:3")
            await adapter._ao_receber_botao(update, _fake_context())

        asyncio.run(cenario())

        assert capturado["escolaridade_escolhida"] == "superior"
        assert capturado["texto"] == "Já fiz uma faculdade"

    def test_botao_de_escolaridade_stale_nao_forca_escolaridade_quando_fase_mudou(self, fake_redis, monkeypatch):
        monkeypatch.setenv("TELEGRAM_BOT_TOKEN", "dummy:token-para-teste")
        capturado = {}

        def fake_responder(user_id, texto, sessao, nivel_escolhido=None, escolaridade_escolhida=None):
            capturado["escolaridade_escolhida"] = escolaridade_escolhida
            return "ok"

        adapter = TelegramAdapter(responder=fake_responder)

        async def cenario():
            await _salvar_sessao_inicial("62", _sessao(dict(_PERFIL_COMPLETO), fase="completo"))
            update = _fake_callback_update(user_id=62, data="escolaridade:0")
            await adapter._ao_receber_botao(update, _fake_context())

        asyncio.run(cenario())

        assert capturado["escolaridade_escolhida"] is None

    @pytest.mark.parametrize(
        "indice,rotulo_esperado,valor_esperado",
        [
            (0, "Ensino fundamental", "ensino fundamental"),
            (1, "Ensino médio", "ensino medio"),
            (2, "Ensino médio técnico", "ensino medio tecnico"),
            (3, "Já fiz uma faculdade", "superior"),
        ],
    )
    def test_cada_botao_de_escolaridade_extrai_o_par_certo(
        self, fake_redis, monkeypatch, indice, rotulo_esperado, valor_esperado
    ):
        monkeypatch.setenv("TELEGRAM_BOT_TOKEN", "dummy:token-para-teste")
        capturado = {}

        def fake_responder(user_id, texto, sessao, nivel_escolhido=None, escolaridade_escolhida=None):
            capturado["texto"] = texto
            capturado["escolaridade_escolhida"] = escolaridade_escolhida
            return "ok"

        adapter = TelegramAdapter(responder=fake_responder)

        async def cenario():
            perfil = {**_PERFIL_SO_FALTA_NIVEL, "escolaridade": None, "nivel": None}
            await _salvar_sessao_inicial("63", _sessao(perfil))
            update = _fake_callback_update(user_id=63, data=f"escolaridade:{indice}")
            await adapter._ao_receber_botao(update, _fake_context())

        asyncio.run(cenario())

        assert capturado["escolaridade_escolhida"] == valor_esperado
        assert capturado["texto"] == rotulo_esperado


class TestCliqueEmBotaoDeEscolaridadeEProduzNivelCoerente:
    """Fim a fim (callback -> `channels.engine.responder` real, sem
    mock) -- confirma que clicar 'Já fiz uma faculdade' nunca deixa a
    pessoa numa situação em que o próximo teclado de nível ofereça algo
    incoerente (ex.: 'Técnico integrado', pensado pra quem ainda vai
    cursar o ensino médio)."""

    def test_apos_clicar_ja_fiz_uma_faculdade_nivel_e_fic_e_niveis_incoerentes_nunca_aparecem(
        self, fake_redis, monkeypatch
    ):
        monkeypatch.setenv("TELEGRAM_BOT_TOKEN", "dummy:token-para-teste")
        # Usa o `responder()` real (coerência de verdade), só troca a
        # chamada paga (LLM) por um stub -- "interesse" segue faltando
        # de propósito, pra continuar em coleta e não escorregar pra
        # `gerar_recomendacao` (que chamaria Anthropic de verdade).
        monkeypatch.setattr(engine, "_gerar_pergunta_coleta", lambda perfil: "Qual área te interessa?")
        adapter = TelegramAdapter(responder=engine.responder)

        async def cenario():
            perfil = {
                "cidade": "Blumenau",
                "escolaridade": None,
                "interesse": None,
                "nivel": None,
                "modalidade": None,
                "alcance": "regional",
            }
            await _salvar_sessao_inicial("64", _sessao(perfil))
            update = _fake_callback_update(user_id=64, data="escolaridade:3")  # "Já fiz uma faculdade"
            await adapter._ao_receber_botao(update, _fake_context())

            from channels.session import carregar_sessao

            return await carregar_sessao("64")

        sessao_final = asyncio.run(cenario())

        assert sessao_final["perfil"]["escolaridade"] == "superior"
        assert sessao_final["perfil"]["nivel"] == "FIC", (
            "escolaridade 'superior' só é compatível com FIC -- deveria ter "
            "pulado a pergunta de nível e preenchido direto"
        )

        botoes_de_nivel_se_perguntasse = engine._botoes_nivel(sessao_final["perfil"]["escolaridade"])
        rotulos_oferecidos = {b.rotulo for linha in botoes_de_nivel_se_perguntasse for b in linha}
        assert "Técnico integrado" not in rotulos_oferecidos
        assert "Técnico subsequente" not in rotulos_oferecidos


class TestDedupDeCliqueDeBotao:
    def test_clique_duplicado_no_mesmo_botao_e_ignorado(self, fake_redis, monkeypatch):
        monkeypatch.setenv("TELEGRAM_BOT_TOKEN", "dummy:token-para-teste")
        chamadas = []

        def fake_responder(user_id, texto, sessao, nivel_escolhido=None, escolaridade_escolhida=None):
            chamadas.append(nivel_escolhido)
            return "ok"

        adapter = TelegramAdapter(responder=fake_responder)

        async def cenario():
            await _salvar_sessao_inicial("65", _sessao(dict(_PERFIL_SO_FALTA_NIVEL)))
            context = _fake_context()
            # Duas mensagens de callback pro MESMO botão, simulando um
            # duplo toque rápido (ex.: rede lenta) -- só a primeira
            # deve de fato chamar o motor de resposta.
            await adapter._ao_receber_botao(_fake_callback_update(user_id=65, data="nivel:2"), context)
            await adapter._ao_receber_botao(_fake_callback_update(user_id=65, data="nivel:2"), context)

        asyncio.run(cenario())

        assert len(chamadas) == 1, f"esperava 1 chamada ao motor, teve {len(chamadas)}: {chamadas}"

    def test_cliques_em_botoes_diferentes_nao_sao_tratados_como_duplicados(self, fake_redis, monkeypatch):
        monkeypatch.setenv("TELEGRAM_BOT_TOKEN", "dummy:token-para-teste")
        chamadas = []

        def fake_responder(user_id, texto, sessao, nivel_escolhido=None, escolaridade_escolhida=None):
            chamadas.append(nivel_escolhido)
            return "ok"

        adapter = TelegramAdapter(responder=fake_responder)

        async def cenario():
            await _salvar_sessao_inicial("66", _sessao(dict(_PERFIL_SO_FALTA_NIVEL)))
            context = _fake_context()
            await adapter._ao_receber_botao(_fake_callback_update(user_id=66, data="nivel:0"), context)
            await adapter._ao_receber_botao(_fake_callback_update(user_id=66, data="nivel:1"), context)

        asyncio.run(cenario())

        assert chamadas == ["tecnico integrado", "tecnico subsequente"]


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


class TestBotoesDeInicio:
    def test_botao_buscar_curso_usa_texto_sintetico_buscar(self, fake_redis, monkeypatch):
        monkeypatch.setenv("TELEGRAM_BOT_TOKEN", "dummy:token-para-teste")
        capturado = {}

        def fake_responder(user_id, texto, sessao, nivel_escolhido=None):
            capturado["texto"] = texto
            return "Em qual cidade você mora?"

        adapter = TelegramAdapter(responder=fake_responder)

        async def cenario():
            await _salvar_sessao_inicial("61", {"perfil": {}, "fase_dialogo": "inicio", "historico": []})
            update = _fake_callback_update(user_id=61, data=engine.CALLBACK_INICIO_BUSCAR)
            await adapter._ao_receber_botao(update, _fake_context())

        asyncio.run(cenario())

        assert capturado["texto"] == "quero buscar um curso"

    def test_botao_tenho_duvida_usa_texto_sintetico_duvida(self, fake_redis, monkeypatch):
        monkeypatch.setenv("TELEGRAM_BOT_TOKEN", "dummy:token-para-teste")
        capturado = {}

        def fake_responder(user_id, texto, sessao, nivel_escolhido=None):
            capturado["texto"] = texto
            return "Pode perguntar!"

        adapter = TelegramAdapter(responder=fake_responder)

        async def cenario():
            await _salvar_sessao_inicial("62", {"perfil": {}, "fase_dialogo": "inicio", "historico": []})
            update = _fake_callback_update(user_id=62, data=engine.CALLBACK_INICIO_DUVIDA)
            await adapter._ao_receber_botao(update, _fake_context())

        asyncio.run(cenario())

        assert capturado["texto"] == "tenho uma duvida"


class TestComandoStart:
    def test_start_mostra_o_menu_inicial(self, fake_redis, monkeypatch):
        monkeypatch.setenv("TELEGRAM_BOT_TOKEN", "dummy:token-para-teste")
        adapter = TelegramAdapter(responder=lambda user_id, texto, sessao, nivel_escolhido=None: "não deveria ser chamado")

        update = _fake_update_comando(user_id=63)
        asyncio.run(adapter._ao_receber_start(update, _fake_context()))

        update.message.reply_text.assert_awaited_once()
        args, kwargs = update.message.reply_text.call_args
        assert args[0] == engine._MENSAGEM_MENU_INICIAL
        teclado = kwargs["reply_markup"]
        assert [b.text for linha in teclado.inline_keyboard for b in linha] == ["Buscar um curso", "Tenho uma dúvida"]

    def test_start_nao_apaga_perfil_existente_so_reabre_a_bifurcacao(self, fake_redis, monkeypatch):
        monkeypatch.setenv("TELEGRAM_BOT_TOKEN", "dummy:token-para-teste")
        adapter = TelegramAdapter(responder=lambda user_id, texto, sessao, nivel_escolhido=None: "ok")

        async def cenario():
            await _salvar_sessao_inicial("64", _sessao(dict(_PERFIL_COMPLETO), fase="completo"))
            update = _fake_update_comando(user_id=64)
            await adapter._ao_receber_start(update, _fake_context())

            from channels.session import carregar_sessao

            return await carregar_sessao("64")

        sessao_depois = asyncio.run(cenario())

        assert sessao_depois["fase_dialogo"] == "inicio"
        assert sessao_depois["perfil"] == _PERFIL_COMPLETO


class TestComandoRecomecar:
    def test_recomecar_aciona_o_mesmo_gatilho_de_reinicio_do_texto(self, fake_redis, monkeypatch):
        monkeypatch.setenv("TELEGRAM_BOT_TOKEN", "dummy:token-para-teste")
        capturado = {}

        def fake_responder(user_id, texto, sessao, nivel_escolhido=None, escolaridade_escolhida=None, alcance_escolhido=None):
            capturado["texto"] = texto
            return "ok"

        adapter = TelegramAdapter(responder=fake_responder)
        update = _fake_update_comando(user_id=70)

        async def cenario():
            await _salvar_sessao_inicial("70", _sessao(dict(_PERFIL_COMPLETO), fase="completo"))
            await adapter._ao_receber_comando_recomecar(update, _fake_context())

        asyncio.run(cenario())

        # Mesmo texto-gatilho que "recomeçar" digitado -- não um texto
        # sintético novo, pra reaproveitar exatamente o mesmo
        # reconhecimento (`dialogue.reset.eh_gatilho_explicito_de_reinicio_total`).
        assert capturado["texto"] == "recomeçar"
        update.message.reply_text.assert_awaited_once()
