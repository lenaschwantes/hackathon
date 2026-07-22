"""Testes do calendário de inscrições (segunda fonte estruturada, sem RAG).

Rodam sem infra (sem Weaviate, sem Voyage, sem docker).

    python -m pytest tests/test_calendario.py -v
"""

import json
from datetime import date

from recommend.calendario import DATA_PATH, Janela, carregar_calendario, consultar_calendario


def _janela(
    nivel: str = "FIC",
    forma_ingresso: str = "inscrição",
    semestre_letivo: str = "2027.1",
    inicio: str | None = "2026-07-01",
    fim: str | None = "2026-07-20",
    data_confirmada: bool = True,
    observacao: str | None = None,
) -> Janela:
    return Janela(
        nivel=nivel,
        forma_ingresso=forma_ingresso,
        semestre_letivo=semestre_letivo,
        inicio=inicio,
        fim=fim,
        data_confirmada=data_confirmada,
        observacao=observacao,
    )


class TestConsultarCalendario:
    def test_janela_aberta_hoje_e_encontrada(self):
        catalogo = [_janela(nivel="superior", inicio="2026-07-01", fim="2026-07-20")]
        resultado = consultar_calendario("superior", date(2026, 7, 10), janelas=catalogo)

        assert resultado["abertas_agora"] == catalogo
        assert resultado["proxima"] is None

    def test_sem_janela_aberta_retorna_a_proxima_futura(self):
        catalogo = [_janela(nivel="superior", inicio="2026-08-01", fim="2026-08-20")]
        resultado = consultar_calendario("superior", date(2026, 7, 10), janelas=catalogo)

        assert resultado["abertas_agora"] == []
        assert resultado["proxima"] == catalogo[0]

    def test_empate_na_data_desempata_pela_ordem_do_catalogo(self):
        primeira = _janela(nivel="superior", forma_ingresso="vestibular", inicio="2026-08-01", fim="2026-08-10")
        segunda = _janela(nivel="superior", forma_ingresso="Sisu", inicio="2026-08-01", fim="2026-08-15")
        resultado = consultar_calendario("superior", date(2026, 7, 10), janelas=[primeira, segunda])

        assert resultado["proxima"].forma_ingresso == "vestibular"

    def test_nivel_sem_calendario_cadastrado_nao_quebra(self):
        catalogo = [_janela(nivel="superior")]
        resultado = consultar_calendario("técnico integrado", date(2026, 7, 10), janelas=catalogo)

        assert resultado == {"abertas_agora": [], "proxima": None, "a_confirmar": []}

    def test_nivel_none_nao_quebra_e_nao_consulta_catalogo(self):
        resultado = consultar_calendario(None, date(2026, 7, 10))
        assert resultado == {"abertas_agora": [], "proxima": None, "a_confirmar": []}

    def test_data_a_confirmar_nao_vira_data_falsa(self):
        catalogo = [
            _janela(
                nivel="superior",
                forma_ingresso="Sisu",
                inicio=None,
                fim=None,
                data_confirmada=False,
                observacao="Data a confirmar conforme cronograma do MEC",
            )
        ]
        resultado = consultar_calendario("superior", date(2026, 7, 10), janelas=catalogo)

        assert resultado["abertas_agora"] == []
        assert resultado["proxima"] is None
        assert len(resultado["a_confirmar"]) == 1
        assert resultado["a_confirmar"][0].inicio is None
        assert resultado["a_confirmar"][0].fim is None
        assert resultado["a_confirmar"][0].forma_ingresso == "Sisu"

    def test_janela_a_confirmar_nunca_conta_como_aberta_nem_proxima(self):
        catalogo = [
            _janela(nivel="superior", forma_ingresso="Sisu", inicio=None, fim=None, data_confirmada=False),
            _janela(nivel="superior", forma_ingresso="vestibular", inicio="2026-09-01", fim="2026-09-20"),
        ]
        resultado = consultar_calendario("superior", date(2026, 7, 10), janelas=catalogo)

        assert len(resultado["a_confirmar"]) == 1
        assert resultado["proxima"].forma_ingresso == "vestibular"

    def test_normalizacao_de_nivel_ignora_caixa_e_acento(self):
        catalogo = [_janela(nivel="técnico integrado", inicio="2026-07-01", fim="2026-07-20")]
        resultado = consultar_calendario("TECNICO INTEGRADO", date(2026, 7, 10), janelas=catalogo)

        assert resultado["abertas_agora"] == catalogo


class TestCarregarCalendario:
    def test_registro_malformado_e_pulado(self, tmp_path):
        bruto = [
            {
                "nivel": "superior",
                "forma_ingresso": "vestibular",
                "semestre_letivo": "2027.1",
                "inicio": "2026-09-01",
                "fim": "2026-09-20",
                "data_confirmada": True,
            },
            {
                "nivel": "superior",
                "forma_ingresso": "vestibular",
                "semestre_letivo": "2027.1",
                "inicio": "data-invalida",
                "fim": "2026-09-20",
                "data_confirmada": True,
            },
            {
                # data_confirmada False mas com datas preenchidas -- inconsistente.
                "nivel": "superior",
                "forma_ingresso": "Sisu",
                "semestre_letivo": "2027.1",
                "inicio": "2026-09-01",
                "fim": "2026-09-20",
                "data_confirmada": False,
            },
            {
                # data_confirmada True mas sem datas -- inconsistente.
                "nivel": "superior",
                "forma_ingresso": "Sisu",
                "semestre_letivo": "2027.1",
                "data_confirmada": True,
            },
        ]
        caminho = tmp_path / "calendario.json"
        caminho.write_text(json.dumps(bruto), encoding="utf-8")

        janelas = carregar_calendario(caminho)

        assert len(janelas) == 1
        assert janelas[0].forma_ingresso == "vestibular"

    def test_catalogo_real_carrega_sem_erros(self):
        janelas = carregar_calendario(DATA_PATH)
        assert len(janelas) >= 20
