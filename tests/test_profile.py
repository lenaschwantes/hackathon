"""
Testes puros do modulo de perfil -- nao tocam Redis, Anthropic nem
nenhuma infraestrutura externa. A chamada ao LLM e isolada em
`_chamar_llm`, entao os testes de extracao usam monkeypatch nela.
"""

from dialogue import profile
from dialogue.profile import Perfil, determinar_fase, extrair_perfil, perfil_vazio


def test_perfil_vazio_esta_incompleto():
    perfil = Perfil(**perfil_vazio())
    assert not perfil.campos_essenciais_completos()
    assert determinar_fase(perfil) == "coletando"


def test_campo_ausente_continua_coletando():
    perfil = Perfil(cidade="Florianopolis", escolaridade="ensino medio completo")
    assert not perfil.campos_essenciais_completos()
    assert determinar_fase(perfil) == "coletando"
    assert perfil.campos_faltantes() == ["interesse", "nivel"]


def test_perfil_completo_muda_de_fase():
    perfil = Perfil(
        cidade="Joinville",
        escolaridade="ensino medio completo",
        interesse="informatica",
        nivel="tecnico integrado",
    )
    assert perfil.campos_essenciais_completos()
    assert determinar_fase(perfil) == "completo"


def test_extrai_campo_de_fala_natural(monkeypatch):
    def fake_chamar_llm(texto, perfil_atual, historico=None):
        return {"cidade": "Lages", "escolaridade": None, "interesse": None, "nivel": None, "modalidade": None}

    monkeypatch.setattr(profile, "_chamar_llm", fake_chamar_llm)
    resultado = extrair_perfil("moro em Lages", perfil_vazio())

    assert resultado.cidade == "Lages"
    assert resultado.escolaridade is None


def test_extrai_nivel_de_curso(monkeypatch):
    def fake_chamar_llm(texto, perfil_atual, historico=None):
        return {"cidade": None, "escolaridade": None, "interesse": None, "nivel": "superior", "modalidade": None}

    monkeypatch.setattr(profile, "_chamar_llm", fake_chamar_llm)
    resultado = extrair_perfil("quero um curso superior", perfil_vazio())

    assert resultado.nivel == "superior"


def test_extracao_nao_apaga_campo_ja_preenchido(monkeypatch):
    def fake_chamar_llm(texto, perfil_atual, historico=None):
        return {"cidade": None, "escolaridade": None, "interesse": "tecnologia", "nivel": None, "modalidade": None}

    monkeypatch.setattr(profile, "_chamar_llm", fake_chamar_llm)
    perfil_atual = {
        "cidade": "Joinville",
        "escolaridade": "ensino medio completo",
        "interesse": None,
        "nivel": None,
        "modalidade": None,
    }
    resultado = extrair_perfil("quero fazer algo com tecnologia", perfil_atual)

    assert resultado.cidade == "Joinville"
    assert resultado.escolaridade == "ensino medio completo"
    assert resultado.interesse == "tecnologia"


def test_extracao_com_falha_no_llm_preserva_perfil(monkeypatch):
    def fake_chamar_llm_com_erro(texto, perfil_atual, historico=None):
        raise RuntimeError("Anthropic indisponivel")

    monkeypatch.setattr(profile, "_chamar_llm", fake_chamar_llm_com_erro)
    perfil_atual = {"cidade": "Blumenau", "escolaridade": None, "interesse": None, "nivel": None, "modalidade": None}
    resultado = extrair_perfil("oi", perfil_atual)

    assert resultado.cidade == "Blumenau"


def test_extracao_usa_historico_pra_resolver_referencia(monkeypatch):
    """
    Se a ultima mensagem do bot (no historico) perguntou o interesse e
    a pessoa responde so "advogado", o LLM (real, aqui mockado) recebe
    o historico como contexto pra entender que isso preenche
    "interesse" -- este teste so confirma que o historico chega
    direitinho na chamada, a interpretacao em si e responsabilidade do
    LLM de verdade.
    """
    capturado = {}

    def fake_chamar_llm(texto, perfil_atual, historico=None):
        capturado["historico"] = historico
        return {"cidade": None, "escolaridade": None, "interesse": "direito", "nivel": None, "modalidade": None}

    monkeypatch.setattr(profile, "_chamar_llm", fake_chamar_llm)
    historico = [
        {"de": "usuario", "texto": "moro em Blumenau"},
        {"de": "bot", "texto": "Qual e o seu interesse?"},
    ]
    resultado = extrair_perfil("advogado", perfil_vazio(), historico=historico)

    assert capturado["historico"] == historico
    assert resultado.interesse == "direito"