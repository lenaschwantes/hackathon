"""Testes de comportamento (LLM real) do tom da coleta de perfil.

Reproduz a sequência real reportada como bug: escolaridade -> interesse
-> alcance, com perfil já parcialmente preenchido em cada turno (nunca
o primeiro turno da conversa) -- confirma que o modelo não reabre
cada resposta com cortesia repetida ("Que bom saber que...").
"""

import os

import pytest
from dotenv import load_dotenv

from dialogue.profile import Perfil

load_dotenv()

pytestmark = pytest.mark.integration

_FRASES_BANIDAS = (
    "que bom",
    "otimo",
    "ótimo",
    "perfeito",
    "obrigad",
    "entendi",
    "anotado",
    "bacana",
    "show",
    "massa",
)


@pytest.fixture
def anthropic_disponivel():
    if not os.environ.get("ANTHROPIC_API_KEY"):
        pytest.skip("Teste de integração pulado: ANTHROPIC_API_KEY não configurada no .env")


def _sem_cortesia(texto: str) -> list[str]:
    texto_lower = texto.lower()
    return [frase for frase in _FRASES_BANIDAS if frase in texto_lower]


class TestTomColetaSemRepeticao:
    def test_sequencia_escolaridade_interesse_alcance_e_direta(self, anthropic_disponivel):
        from channels.engine import _gerar_pergunta_coleta

        turnos = [
            Perfil(cidade="Blumenau"),
            Perfil(cidade="Blumenau", escolaridade="superior"),
            Perfil(cidade="Blumenau", escolaridade="superior", interesse="tecnologia"),
        ]
        respostas = [_gerar_pergunta_coleta(perfil) for perfil in turnos]

        for nome, resposta in zip(("escolaridade", "interesse", "alcance"), respostas):
            bateu = _sem_cortesia(resposta)
            assert not bateu, (
                f"turno {nome!r} reabriu com cortesia banida {bateu} -- resposta: {resposta!r}"
            )

        # Turnos consecutivos não devem repetir a mesma estrutura de abertura.
        aberturas = [r.strip().split()[:3] for r in respostas]
        for i in range(1, len(aberturas)):
            assert aberturas[i] != aberturas[i - 1], (
                f"abertura do turno {i} repete a do turno {i - 1}: {aberturas[i]}"
            )
