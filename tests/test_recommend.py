"""Testes do motor de recomendação (filtro estruturado, sem RAG).

Rodam sem infra (sem Weaviate, sem Voyage, sem docker).

    python -m pytest tests/test_recommend.py -v
"""

import json
from datetime import date

from recommend.opportunities import (
    DATA_PATH,
    Oportunidade,
    carregar_oportunidades,
    recomendar,
)


def _op(
    cidade: str = "Blumenau",
    modalidade: str = "presencial",
    nivel: str = "FIC",
    forma_ingresso: str = "análise",
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
        forma_ingresso=forma_ingresso,
        inscricao_inicio=inicio,
        inscricao_fim=fim,
        link_edital="https://example.org/edital",
    )


class TestRecomendar:
    def test_inscricao_aberta_encontrada(self):
        catalogo = [_op(cidade="Blumenau", inicio="2026-07-01", fim="2026-07-20")]
        resultado = recomendar("Blumenau", date(2026, 7, 10), oportunidades=catalogo)
        assert resultado["abertas"] == catalogo
        assert resultado["proxima"] is None

    def test_nenhuma_aberta_mas_existe_proxima(self):
        catalogo = [_op(cidade="Blumenau", inicio="2026-08-01", fim="2026-08-20")]
        resultado = recomendar("Blumenau", date(2026, 7, 10), oportunidades=catalogo)
        assert resultado["abertas"] == []
        assert resultado["proxima"] == catalogo[0]

    def test_cidade_sem_oferta_presencial_cai_no_ead(self):
        # Nenhuma oferta presencial em Chapecó, mas a EAD dispensa geografia.
        catalogo = [
            _op(cidade="Blumenau", modalidade="presencial", inicio="2026-01-01", fim="2026-01-10"),
            _op(cidade="Florianópolis", modalidade="EAD", inicio="2026-07-01", fim="2026-07-20"),
        ]
        resultado = recomendar("Chapecó", date(2026, 7, 10), oportunidades=catalogo)
        assert len(resultado["abertas"]) == 1
        assert resultado["abertas"][0].modalidade == "EAD"

    def test_empate_na_data_da_proxima_desempata_pela_ordem_do_catalogo(self):
        primeira = _op(cidade="Blumenau", curso="Primeira", inicio="2026-08-01", fim="2026-08-10")
        segunda = _op(cidade="Blumenau", curso="Segunda", inicio="2026-08-01", fim="2026-08-15")
        resultado = recomendar("Blumenau", date(2026, 7, 10), oportunidades=[primeira, segunda])
        assert resultado["proxima"].curso == "Primeira"

    def test_hoje_na_borda_de_inicio_e_de_fim(self):
        oportunidade = _op(cidade="Blumenau", inicio="2026-07-01", fim="2026-07-20")

        resultado_inicio = recomendar("Blumenau", date(2026, 7, 1), oportunidades=[oportunidade])
        resultado_fim = recomendar("Blumenau", date(2026, 7, 20), oportunidades=[oportunidade])

        assert resultado_inicio["abertas"] == [oportunidade]
        assert resultado_fim["abertas"] == [oportunidade]

    def test_normalizacao_de_cidade_ignora_caixa_e_acento(self):
        catalogo = [_op(cidade="São José", inicio="2026-07-01", fim="2026-07-20")]
        resultado = recomendar("sao jose", date(2026, 7, 10), oportunidades=catalogo)
        assert resultado["abertas"] == catalogo


class TestCarregarOportunidades:
    def test_registro_malformado_e_pulado(self, tmp_path):
        bruto = [
            {
                "curso": "Curso Válido",
                "campus": "Campus X",
                "cidade": "Blumenau",
                "modalidade": "presencial",
                "nivel": "FIC",
                "forma_ingresso": "análise",
                "inscricao_inicio": "2026-07-01",
                "inscricao_fim": "2026-07-20",
                "link_edital": "https://example.org/edital-valido",
            },
            {
                "curso": "Curso Malformado",
                "campus": "Campus Y",
                "cidade": "Blumenau",
                "modalidade": "presencial",
                "nivel": "FIC",
                "forma_ingresso": "análise",
                "inscricao_inicio": "data-invalida",
                "inscricao_fim": "2026-07-20",
                "link_edital": "https://example.org/edital-invalido",
            },
        ]
        caminho = tmp_path / "opportunities.json"
        caminho.write_text(json.dumps(bruto), encoding="utf-8")

        oportunidades = carregar_oportunidades(caminho)

        assert len(oportunidades) == 1
        assert oportunidades[0].curso == "Curso Válido"

    def test_catalogo_real_carrega_sem_erros(self):
        oportunidades = carregar_oportunidades(DATA_PATH)
        assert 15 <= len(oportunidades) <= 20
