"""
Testes puros do modulo de perfil -- nao tocam Redis, Anthropic nem
nenhuma infraestrutura externa. A chamada ao LLM e isolada em
`_chamar_llm`, entao os testes de extracao usam monkeypatch nela.
"""

from dialogue import profile
from dialogue.profile import OPCOES_NIVEL, Perfil, determinar_fase, extrair_perfil, perfil_vazio


def test_perfil_vazio_esta_incompleto():
    perfil = Perfil(**perfil_vazio())
    assert not perfil.campos_essenciais_completos()
    assert determinar_fase(perfil) == "coletando"


def test_opcoes_nivel_tem_quatro_entradas_com_valores_canonicos():
    assert len(OPCOES_NIVEL) == 4
    valores = [valor for _, valor in OPCOES_NIVEL]
    assert valores == ["tecnico integrado", "tecnico subsequente", "superior", "FIC"]
    assert all(rotulo for rotulo, _ in OPCOES_NIVEL)


def test_campo_ausente_continua_coletando():
    perfil = Perfil(cidade="Florianopolis", escolaridade="ensino medio completo")
    assert not perfil.campos_essenciais_completos()
    assert determinar_fase(perfil) == "coletando"
    assert perfil.campos_faltantes() == ["interesse", "alcance", "nivel"]


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


def test_extrai_alcance_local_de_fala_natural(monkeypatch):
    def fake_chamar_llm(texto, perfil_atual, historico=None):
        return {
            "cidade": None, "escolaridade": None, "interesse": None, "nivel": None,
            "modalidade": None, "alcance": "local",
        }

    monkeypatch.setattr(profile, "_chamar_llm", fake_chamar_llm)
    resultado = extrair_perfil("so aqui na minha cidade mesmo", perfil_vazio())

    assert resultado.alcance == "local"


def test_extrai_alcance_regional_de_fala_natural(monkeypatch):
    def fake_chamar_llm(texto, perfil_atual, historico=None):
        return {
            "cidade": None, "escolaridade": None, "interesse": None, "nivel": None,
            "modalidade": None, "alcance": "regional",
        }

    monkeypatch.setattr(profile, "_chamar_llm", fake_chamar_llm)
    resultado = extrair_perfil("posso ir pra Florianopolis, topo ir pra perto", perfil_vazio())

    assert resultado.alcance == "regional"


def test_extrai_alcance_ead_de_fala_natural(monkeypatch):
    def fake_chamar_llm(texto, perfil_atual, historico=None):
        return {
            "cidade": None, "escolaridade": None, "interesse": None, "nivel": None,
            "modalidade": None, "alcance": "ead",
        }

    monkeypatch.setattr(profile, "_chamar_llm", fake_chamar_llm)
    resultado = extrair_perfil("prefiro a distancia, nao posso me deslocar", perfil_vazio())

    assert resultado.alcance == "ead"


def test_extrai_alcance_qualquer_de_fala_natural(monkeypatch):
    def fake_chamar_llm(texto, perfil_atual, historico=None):
        return {
            "cidade": None, "escolaridade": None, "interesse": None, "nivel": None,
            "modalidade": None, "alcance": "qualquer",
        }

    monkeypatch.setattr(profile, "_chamar_llm", fake_chamar_llm)
    resultado = extrair_perfil("tanto faz onde", perfil_vazio())

    assert resultado.alcance == "qualquer"


def test_alcance_nao_bloqueia_perfil_completo():
    perfil = Perfil(
        cidade="Joinville",
        escolaridade="ensino medio completo",
        interesse="informatica",
        nivel="tecnico integrado",
    )
    assert perfil.campos_essenciais_completos()
    assert determinar_fase(perfil) == "completo"
    assert perfil.alcance is None


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


def test_regressao_nao_sei_depois_area_real_avanca_o_perfil(monkeypatch):
    """
    Reproduz o dialogo real reportado: "Ainda não sei" e depois
    "Saúde", com cidade/escolaridade/nivel ja preenchidos (so falta
    interesse). A primeira mensagem nao deve preencher interesse (o
    LLM corretamente nao inventa uma area); a segunda deve preencher
    "Saúde" normalmente e completar o perfil.
    """
    perfil_atual = {
        "cidade": "Blumenau", "escolaridade": "ensino medio completo",
        "interesse": None, "nivel": "tecnico integrado", "modalidade": None, "alcance": None,
    }
    historico = [
        {"de": "usuario", "texto": "moro em blumenau, terminei o ensino medio"},
        {"de": "bot", "texto": "Legal! Me conta, qual área te interessa?"},
    ]

    def fake_chamar_llm_sem_area(texto, perfil_atual, historico=None):
        return {**perfil_atual, "interesse": None, "modalidade": None, "alcance": None}

    monkeypatch.setattr(profile, "_chamar_llm", fake_chamar_llm_sem_area)
    resultado_1 = extrair_perfil("Ainda não sei", perfil_atual, historico=historico)

    assert resultado_1.interesse is None
    assert determinar_fase(resultado_1) == "coletando"

    historico.append({"de": "usuario", "texto": "Ainda não sei"})
    historico.append({"de": "bot", "texto": "Sem problema! Pode ser algo bem amplo, tipo saúde, tecnologia..."})

    def fake_chamar_llm_com_saude(texto, perfil_atual, historico=None):
        return {**perfil_atual, "interesse": "Saúde", "modalidade": None, "alcance": None}

    monkeypatch.setattr(profile, "_chamar_llm", fake_chamar_llm_com_saude)
    resultado_2 = extrair_perfil("Saúde", resultado_1.model_dump(), historico=historico)

    assert resultado_2.interesse == "Saúde"
    assert determinar_fase(resultado_2) == "completo"


def test_insistencia_dupla_em_nao_sei_avanca_sem_area_especifica(monkeypatch):
    perfil_atual = {
        "cidade": "Blumenau", "escolaridade": "ensino medio completo",
        "interesse": None, "nivel": "tecnico integrado", "modalidade": None, "alcance": None,
    }

    def fake_chamar_llm_sem_area(texto, perfil_atual, historico=None):
        return {**perfil_atual, "interesse": None, "modalidade": None, "alcance": None}

    monkeypatch.setattr(profile, "_chamar_llm", fake_chamar_llm_sem_area)

    historico = [{"de": "bot", "texto": "Qual área te interessa?"}]
    resultado_1 = extrair_perfil("não sei", perfil_atual, historico=historico)
    assert resultado_1.interesse is None

    historico.append({"de": "usuario", "texto": "não sei"})
    historico.append({"de": "bot", "texto": "Sem problema! Pode ser algo bem amplo..."})
    resultado_2 = extrair_perfil("não sei", resultado_1.model_dump(), historico=historico)

    assert resultado_2.interesse == "sem preferência definida"
    assert determinar_fase(resultado_2) == "completo"


def test_extracao_nao_apaga_alcance_ja_preenchido(monkeypatch):
    def fake_chamar_llm(texto, perfil_atual, historico=None):
        return {
            "cidade": None, "escolaridade": None, "interesse": "tecnologia", "nivel": None,
            "modalidade": None, "alcance": None,
        }

    monkeypatch.setattr(profile, "_chamar_llm", fake_chamar_llm)
    perfil_atual = {
        "cidade": "Joinville", "escolaridade": "ensino medio completo", "interesse": None,
        "nivel": None, "modalidade": None, "alcance": "regional",
    }
    resultado = extrair_perfil("quero fazer algo com tecnologia", perfil_atual)

    assert resultado.alcance == "regional"


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