"""Testes estáticos de regra sobre o conteúdo dos prompts.

Não chamam o LLM -- garantem só que as regras de tom que já provamos
funcionar (ver `tests/integration/test_tom_coleta.py`) continuam
escritas no prompt, pra pegar quem remover a regra sem perceber.
"""

from config.prompts import PROMPT_COLETA


class TestPromptColetaSemCortesiaRepetida:
    def test_proibe_frases_de_efeito_reportadas_no_bug(self):
        texto = PROMPT_COLETA.lower()
        for frase in ("que bom saber", "que bom saber disso, obrigado"):
            assert frase in texto, f"prompt não lista mais a frase banida: {frase!r}"

    def test_regra_vale_pra_toda_mensagem_nao_so_a_primeira(self):
        texto = PROMPT_COLETA.lower()
        assert "toda mensagem da coleta" in texto
        assert "sem excecao" in texto

    def test_confirmacao_curta_nao_pode_repetir_estrutura(self):
        texto = PROMPT_COLETA.lower()
        assert "nunca repita a mesma" in texto and "estrutura" in texto
