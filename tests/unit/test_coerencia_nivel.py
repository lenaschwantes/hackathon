"""Coerência entre escolaridade e nível de curso oferecido.

Reproduz o bug reportado: escolaridade e nível eram coletados como
campos independentes, então quem já tinha feito faculdade ("superior")
ainda via "Técnico integrado" como opção -- oferta incoerente, já que
técnico integrado é pensado pra quem vai cursar junto com o ensino
médio.
"""

from channels.engine import _botoes_nivel
from dialogue.profile import Perfil, niveis_compativeis


class TestNiveisCompativeis:
    def test_superior_nunca_oferece_tecnico_integrado(self):
        compativeis = niveis_compativeis("superior")
        assert "tecnico integrado" not in compativeis

    def test_fundamental_nao_oferece_superior(self):
        compativeis = niveis_compativeis("ensino fundamental")
        assert "superior" not in compativeis
        assert "tecnico subsequente" not in compativeis

    def test_medio_completo_oferece_subsequente_superior_e_fic(self):
        compativeis = niveis_compativeis("ensino medio")
        assert set(compativeis) == {"tecnico subsequente", "superior", "FIC"}

    def test_escolaridade_desconhecida_nao_restringe(self):
        # Valor fora do vocabulário fechado (ex.: já filtrado por outro
        # caminho) não deve travar a coleta -- sem restrição nesse caso.
        assert niveis_compativeis("mestrado") == niveis_compativeis(None)

    def test_normaliza_acento_e_caixa(self):
        assert niveis_compativeis("Ensino Fundamental") == niveis_compativeis("ensino fundamental")


class TestPerguntaDeNivelSuprimidaOuRestringida:
    def test_superior_tem_um_unico_nivel_plausivel_pula_pergunta(self):
        """Escolaridade com um único nível coerente (superior -> FIC):
        a pergunta de nível é suprimida -- o próprio `campos_faltantes`
        não deve mais listar "nivel" quando ele já está implícito."""
        perfil = Perfil(
            cidade="Blumenau", escolaridade="superior", interesse="tecnologia", nivel="FIC"
        )
        assert "nivel" not in perfil.campos_faltantes()

    def test_fundamental_restringe_botoes_a_duas_opcoes(self):
        botoes = [b for linha in _botoes_nivel("ensino fundamental") for b in linha]
        rotulos = {b.rotulo for b in botoes}
        assert rotulos == {"Técnico integrado", "FIC (curso rápido)"}

    def test_superior_restringe_botoes_a_uma_opcao(self):
        botoes = [b for linha in _botoes_nivel("superior") for b in linha]
        rotulos = {b.rotulo for b in botoes}
        assert rotulos == {"FIC (curso rápido)"}

    def test_callback_data_preserva_indice_original(self):
        # "FIC" é o índice 3 em OPCOES_NIVEL -- mesmo filtrado pra ser o
        # único botão, o callback precisa continuar apontando pro índice
        # certo, já que `channels/telegram.py` resolve `OPCOES_NIVEL[i]`.
        botoes = [b for linha in _botoes_nivel("superior") for b in linha]
        assert botoes == [botoes[0]]
        assert botoes[0].callback_data == "nivel:3"


class TestExtracaoPreenchNivelPorCoerencia:
    def test_escolaridade_superior_preenche_nivel_fic_automaticamente(self, monkeypatch):
        import dialogue.profile as profile_module

        monkeypatch.setattr(
            profile_module, "_chamar_llm", lambda texto, perfil_atual, historico=None: {
                "escolaridade": "superior"
            }
        )
        perfil = profile_module.extrair_perfil("ja fiz uma faculdade", {"cidade": "Blumenau"})
        assert perfil.nivel == "FIC"

    def test_escolaridade_medio_nao_preenche_nivel_sozinho(self, monkeypatch):
        import dialogue.profile as profile_module

        monkeypatch.setattr(
            profile_module, "_chamar_llm", lambda texto, perfil_atual, historico=None: {
                "escolaridade": "ensino medio"
            }
        )
        perfil = profile_module.extrair_perfil("terminei o ensino medio", {"cidade": "Blumenau"})
        # Mais de um nível plausível (subsequente, superior, FIC) --
        # continua sem "nivel" pra pessoa escolher, só restrito nas opções.
        assert perfil.nivel is None
        assert "nivel" in perfil.campos_faltantes()


class TestReproducaoDialogoRealEscolaridadeSuperior:
    def test_dialogo_ja_fiz_uma_faculdade_nunca_pergunta_tecnico_integrado(self, monkeypatch):
        """Regressão do bug relatado: no diálogo real (escolha da opção 4,
        'Já fiz uma faculdade'), a pergunta seguinte não pode mais
        oferecer 'Técnico integrado' -- nem em texto, nem em botão."""
        import dialogue.profile as profile_module

        monkeypatch.setattr(
            profile_module, "_chamar_llm", lambda texto, perfil_atual, historico=None: {
                "escolaridade": "superior"
            }
        )
        perfil = profile_module.extrair_perfil("4", {"cidade": "Blumenau", "interesse": "saude"})

        assert perfil.nivel == "FIC", "esperava pular a pergunta e inferir FIC"
        assert "nivel" not in perfil.campos_faltantes()

        botoes = [b for linha in _botoes_nivel(perfil.escolaridade) for b in linha]
        assert all(b.rotulo != "Técnico integrado" for b in botoes)
