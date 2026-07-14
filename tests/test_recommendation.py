"""
Testes puros da ponte entre perfil e motor de recomendacao -- nao
tocam Groq nem nenhuma infraestrutura externa. A chamada ao LLM e
isolada em `_chamar_llm`, entao os testes de `gerar_recomendacao`
usam monkeypatch nela, igual `tests/test_profile.py` faz.
"""

from datetime import date

from dialogue import recommendation
from dialogue.profile import Perfil
from dialogue.recommendation import gerar_recomendacao, montar_contexto, quer_nova_recomendacao
from recommend.opportunities import Oportunidade


def _op(
    cidade: str = "Blumenau",
    modalidade: str = "presencial",
    nivel: str = "FIC",
    curso: str = "Curso Teste",
    inicio: str = "2026-01-01",
    fim: str = "2026-01-10",
) -> Oportunidade:
    return Oportunidade(
        curso=curso,
        campus="Campus Teste",
        cidade=cidade,
        modalidade=modalidade,
        nivel=nivel,
        forma_ingresso="análise",
        inscricao_inicio=inicio,
        inscricao_fim=fim,
        link_edital="https://example.org/edital",
    )


def _fake_recomendar(catalogo):
    def fake(cidade, hoje, nivel=None, modalidade=None, *, oportunidades=None):
        from recommend.opportunities import recomendar as recomendar_real

        return recomendar_real(cidade, hoje, nivel, modalidade, oportunidades=catalogo)

    return fake


class TestMontarContexto:
    def test_com_oportunidade_aberta(self, monkeypatch):
        catalogo = [_op(cidade="Blumenau", inicio="2026-07-01", fim="2026-07-20")]
        monkeypatch.setattr(recommendation, "recomendar", _fake_recomendar(catalogo))
        perfil = Perfil(cidade="Blumenau", escolaridade="ensino medio completo", interesse="tecnologia")

        contexto = montar_contexto(perfil, date(2026, 7, 10))

        assert contexto["interesse"] == "tecnologia"
        assert len(contexto["abertas"]) == 1
        assert contexto["abertas"][0]["curso"] == "Curso Teste"
        assert contexto["abertas"][0]["inscricao_inicio"] == "2026-07-01"
        assert contexto["proxima"] is None

    def test_sem_aberta_mas_com_proxima(self, monkeypatch):
        catalogo = [_op(cidade="Blumenau", inicio="2026-08-01", fim="2026-08-20")]
        monkeypatch.setattr(recommendation, "recomendar", _fake_recomendar(catalogo))
        perfil = Perfil(cidade="Blumenau", escolaridade="ensino medio completo", interesse="tecnologia")

        contexto = montar_contexto(perfil, date(2026, 7, 10))

        assert contexto["abertas"] == []
        assert contexto["proxima"]["curso"] == "Curso Teste"
        assert contexto["proxima"]["inscricao_inicio"] == "2026-08-01"

    def test_sem_nenhuma_compativel(self, monkeypatch):
        monkeypatch.setattr(recommendation, "recomendar", _fake_recomendar([]))
        perfil = Perfil(cidade="Cidade Sem Oferta", escolaridade="ensino medio completo", interesse="tecnologia")

        contexto = montar_contexto(perfil, date(2026, 7, 10))

        assert contexto["abertas"] == []
        assert contexto["proxima"] is None

    def test_repassa_nivel_do_perfil_pro_filtro(self, monkeypatch):
        capturado = {}

        def fake_recomendar(cidade, hoje, nivel=None, modalidade=None, *, oportunidades=None):
            capturado["nivel"] = nivel
            return {"abertas": [], "proxima": None}

        monkeypatch.setattr(recommendation, "recomendar", fake_recomendar)
        perfil = Perfil(
            cidade="Blumenau",
            escolaridade="ensino medio completo",
            interesse="tecnologia",
            nivel="superior",
        )

        montar_contexto(perfil, date(2026, 7, 10))

        assert capturado["nivel"] == "superior"


class TestGerarRecomendacao:
    def test_repassa_contexto_correto_pro_llm(self, monkeypatch):
        capturado = {}

        def fake_chamar_llm(contexto):
            capturado["contexto"] = contexto
            return "Encontrei uma opção pra você em Blumenau!"

        catalogo = [_op(cidade="Blumenau", inicio="2026-07-01", fim="2026-07-20")]
        monkeypatch.setattr(recommendation, "_chamar_llm", fake_chamar_llm)
        monkeypatch.setattr(recommendation, "recomendar", _fake_recomendar(catalogo))

        perfil = Perfil(cidade="Blumenau", escolaridade="ensino medio completo", interesse="tecnologia")
        resultado = gerar_recomendacao(perfil, hoje=date(2026, 7, 10))

        assert resultado == "Encontrei uma opção pra você em Blumenau!"
        assert capturado["contexto"]["interesse"] == "tecnologia"
        assert len(capturado["contexto"]["abertas"]) == 1


class TestQuerNovaRecomendacao:
    def test_classificador_diz_que_quer_nova_recomendacao(self, monkeypatch):
        monkeypatch.setattr(
            recommendation, "_chamar_llm_classificador", lambda texto: {"quer_nova_recomendacao": True}
        )
        assert quer_nova_recomendacao("mostra outra opção") is True

    def test_classificador_diz_que_e_pergunta_normal(self, monkeypatch):
        monkeypatch.setattr(
            recommendation, "_chamar_llm_classificador", lambda texto: {"quer_nova_recomendacao": False}
        )
        assert quer_nova_recomendacao("quando fecha a inscrição?") is False

    def test_falha_no_classificador_nao_bloqueia_e_assume_false(self, monkeypatch):
        def _chamar_llm_com_erro(texto):
            raise RuntimeError("Groq indisponível")

        monkeypatch.setattr(recommendation, "_chamar_llm_classificador", _chamar_llm_com_erro)
        assert quer_nova_recomendacao("qualquer coisa") is False
