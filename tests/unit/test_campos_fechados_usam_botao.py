"""Garante que campo de conjunto fechado nunca mistura menu numerado em
texto com o teclado de botões -- bug relatado: a coleta chegou a exibir
"1) Ensino fundamental / 2) Ensino médio / ..." junto do mecanismo de
botão já implementado. Onde cabe botão, as opções vêm só como teclado
(lista de rótulo/valor), nunca embutidas na string da pergunta.
"""

from __future__ import annotations

import re

import pytest

from channels.engine import _BOTOES_ESCOLARIDADE, _botoes_nivel, _com_botoes_de_campo_fechado
from config.prompts import PROMPT_COLETA
from dialogue.profile import Perfil

_PADRAO_MENU_NUMERADO = re.compile(r"^\s*\d\)\s", re.MULTILINE)


class TestPromptNaoDescreveMenuNumerado:
    def test_prompt_coleta_nao_contem_padrao_de_lista_numerada(self):
        assert not _PADRAO_MENU_NUMERADO.search(PROMPT_COLETA), (
            "PROMPT_COLETA ainda instrui um menu tipo '1) ...' em texto -- "
            "campo de conjunto fechado deve usar só o teclado de botões."
        )

    def test_prompt_nao_instrui_responder_com_numero(self):
        texto = PROMPT_COLETA.lower()
        assert "pode responder so o numero" not in texto
        assert "responda com o numero" not in texto


class TestCamposFechadosGeramTecladoDeBotoes:
    """Pra cada campo de conjunto fechado, a pergunta que sai de
    `_com_botoes_de_campo_fechado` vem com `.botoes` -- uma lista de
    rótulo/valor (`Botao`) -- nunca com as opções escritas dentro da
    própria string da pergunta."""

    @pytest.mark.parametrize(
        "perfil_faltando,botoes_esperados",
        [
            (
                Perfil(cidade="Blumenau"),  # falta escolaridade primeiro
                _BOTOES_ESCOLARIDADE,
            ),
            (
                Perfil(cidade="Blumenau", escolaridade="ensino medio", interesse="tecnologia", alcance="regional"),
                None,  # comparado via _botoes_nivel abaixo, depende da escolaridade
            ),
        ],
    )
    def test_pergunta_vem_com_teclado_nao_com_lista_em_texto(self, perfil_faltando, botoes_esperados):
        pergunta_curta = "Pergunta genérica de coleta (sem opções embutidas)."
        resultado = _com_botoes_de_campo_fechado(pergunta_curta, perfil_faltando)

        botoes = getattr(resultado, "botoes", None)
        assert botoes is not None, "campo de conjunto fechado deveria ter vindo com teclado de botões"

        if botoes_esperados is None:
            botoes_esperados = _botoes_nivel(perfil_faltando.escolaridade)
        assert botoes == botoes_esperados

        # A string da pergunta em si nunca precisa (nem deve) enumerar
        # as opções -- quem carrega as opções é `.botoes`.
        assert not _PADRAO_MENU_NUMERADO.search(str(resultado))

    def test_campo_aberto_nao_ganha_teclado(self):
        perfil_faltando_interesse = Perfil(cidade="Blumenau", escolaridade="ensino medio")
        resultado = _com_botoes_de_campo_fechado("Qual área te interessa?", perfil_faltando_interesse)
        assert getattr(resultado, "botoes", None) is None
