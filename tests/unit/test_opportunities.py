"""Testes do motor de recomendação (filtro estruturado, sem RAG).

Rodam sem infra (sem Weaviate, sem Voyage, sem docker).

    python -m pytest tests/test_recommend.py -v
"""

import itertools
import json
from datetime import date

import pytest

from recommend.opportunities import (
    DATA_PATH,
    Oportunidade,
    carregar_oportunidades,
    recomendar,
)

_contador_link = itertools.count()


def _op(
    cidade: str = "Blumenau",
    modalidade: str = "presencial",
    nivel: str = "FIC",
    forma_ingresso: str = "análise",
    curso: str = "Curso Teste",
    inicio: str = "2026-01-01",
    fim: str = "2026-01-10",
    link_edital: str | None = None,
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
        # link único por chamada por padrão -- evita colisão acidental
        # com o dedup por link_edital; testes de dedup passam o mesmo
        # link de propósito.
        link_edital=link_edital or f"https://example.org/edital-{next(_contador_link)}",
    )


class TestRecomendar:
    def test_inscricao_aberta_encontrada(self):
        catalogo = [_op(cidade="Blumenau", inicio="2026-07-01", fim="2026-07-20")]
        resultado = recomendar("Blumenau", date(2026, 7, 10), oportunidades=catalogo)
        assert resultado["na_cidade"] == catalogo
        assert resultado["proxima"] is None

    def test_nenhuma_aberta_mas_existe_proxima(self):
        catalogo = [_op(cidade="Blumenau", inicio="2026-08-01", fim="2026-08-20")]
        resultado = recomendar("Blumenau", date(2026, 7, 10), oportunidades=catalogo)
        assert resultado["na_cidade"] == []
        assert resultado["proxima"] == catalogo[0]

    def test_cidade_sem_oferta_presencial_cai_no_ead(self):
        # Nenhuma oferta presencial em Chapecó, mas a EAD dispensa geografia.
        catalogo = [
            _op(cidade="Blumenau", modalidade="presencial", inicio="2026-01-01", fim="2026-01-10"),
            _op(cidade="Florianópolis", modalidade="EAD", inicio="2026-07-01", fim="2026-07-20"),
        ]
        resultado = recomendar("Chapecó", date(2026, 7, 10), oportunidades=catalogo)
        assert resultado["na_cidade"] == []
        assert resultado["outras_cidades"] == []
        assert len(resultado["ead"]) == 1
        assert resultado["ead"][0].modalidade == "EAD"

    def test_empate_na_data_da_proxima_desempata_pela_ordem_do_catalogo(self):
        primeira = _op(cidade="Blumenau", curso="Primeira", inicio="2026-08-01", fim="2026-08-10")
        segunda = _op(cidade="Blumenau", curso="Segunda", inicio="2026-08-01", fim="2026-08-15")
        resultado = recomendar("Blumenau", date(2026, 7, 10), oportunidades=[primeira, segunda])
        assert resultado["proxima"].curso == "Primeira"

    def test_hoje_na_borda_de_inicio_e_de_fim(self):
        oportunidade = _op(cidade="Blumenau", inicio="2026-07-01", fim="2026-07-20")

        resultado_inicio = recomendar("Blumenau", date(2026, 7, 1), oportunidades=[oportunidade])
        resultado_fim = recomendar("Blumenau", date(2026, 7, 20), oportunidades=[oportunidade])

        assert resultado_inicio["na_cidade"] == [oportunidade]
        assert resultado_fim["na_cidade"] == [oportunidade]

    def test_normalizacao_de_cidade_ignora_caixa_e_acento(self):
        catalogo = [_op(cidade="São José", inicio="2026-07-01", fim="2026-07-20")]
        resultado = recomendar("sao jose", date(2026, 7, 10), oportunidades=catalogo)
        assert resultado["na_cidade"] == catalogo

    def test_dedup_por_link_edital_preserva_ordem(self):
        original = _op(curso="Original", link_edital="https://example.org/mesmo-edital", inicio="2026-07-01", fim="2026-07-20")
        duplicata = _op(curso="Original (duplicata)", link_edital="https://example.org/mesmo-edital", inicio="2026-07-01", fim="2026-07-20")
        resultado = recomendar("Blumenau", date(2026, 7, 10), oportunidades=[original, duplicata])
        assert resultado["na_cidade"] == [original]


class TestRecomendarPorCamadaDeProximidade:
    def _catalogo_com_quatro_camadas(self) -> list[Oportunidade]:
        return [
            _op(cidade="Florianópolis", curso="Na cidade", inicio="2026-07-01", fim="2026-07-20"),
            _op(cidade="São José", curso="Na regiao", inicio="2026-07-01", fim="2026-07-20"),
            _op(cidade="Criciúma", modalidade="EAD", curso="EAD", inicio="2026-07-01", fim="2026-07-20"),
            _op(cidade="Chapecó", curso="Outra cidade", inicio="2026-07-01", fim="2026-07-20"),
        ]

    def test_alcance_local_so_recebe_da_cidade(self):
        resultado = recomendar(
            "Florianópolis", date(2026, 7, 10), alcance="local", oportunidades=self._catalogo_com_quatro_camadas()
        )

        assert [o.curso for o in resultado["na_cidade"]] == ["Na cidade"]
        assert resultado["regiao"] == []
        assert resultado["ead"] == []
        assert resultado["outras_cidades"] == []

    def test_alcance_regional_inclui_a_regiao(self):
        resultado = recomendar(
            "Florianópolis", date(2026, 7, 10), alcance="regional", oportunidades=self._catalogo_com_quatro_camadas()
        )

        assert [o.curso for o in resultado["na_cidade"]] == ["Na cidade"]
        assert [o.curso for o in resultado["regiao"]] == ["Na regiao"]
        assert resultado["ead"] == []
        assert resultado["outras_cidades"] == []

    def test_ead_sempre_aparece_independente_da_cidade_da_pessoa(self):
        catalogo = [_op(cidade="Criciúma", modalidade="EAD", curso="EAD", inicio="2026-07-01", fim="2026-07-20")]

        resultado_florianopolis = recomendar("Florianópolis", date(2026, 7, 10), oportunidades=catalogo)
        resultado_chapeco = recomendar("Chapecó", date(2026, 7, 10), oportunidades=catalogo)

        assert [o.curso for o in resultado_florianopolis["ead"]] == ["EAD"]
        assert [o.curso for o in resultado_chapeco["ead"]] == ["EAD"]

    def test_alcance_qualquer_traz_outras_cidades(self):
        resultado = recomendar(
            "Florianópolis", date(2026, 7, 10), alcance="qualquer", oportunidades=self._catalogo_com_quatro_camadas()
        )

        assert [o.curso for o in resultado["na_cidade"]] == ["Na cidade"]
        assert [o.curso for o in resultado["regiao"]] == ["Na regiao"]
        assert [o.curso for o in resultado["ead"]] == ["EAD"]
        assert [o.curso for o in resultado["outras_cidades"]] == ["Outra cidade"]

    def test_default_e_inclusivo_mas_nao_extrapola_pra_outras_cidades(self):
        resultado = recomendar("Florianópolis", date(2026, 7, 10), oportunidades=self._catalogo_com_quatro_camadas())

        assert [o.curso for o in resultado["na_cidade"]] == ["Na cidade"]
        assert [o.curso for o in resultado["regiao"]] == ["Na regiao"]
        assert [o.curso for o in resultado["ead"]] == ["EAD"]
        assert resultado["outras_cidades"] == []

    def test_fallback_da_proxima_respeita_camadas_aceitas(self):
        # Aberta agora, mas fora do alcance aceito por default (outras
        # cidades) -- não pode vazar pra "proxima". A futura dentro de
        # uma camada aceita (na_cidade) e que deve aparecer.
        fora_do_alcance_aberta = _op(cidade="Chapecó", curso="Fora, aberta", inicio="2026-07-01", fim="2026-07-20")
        na_cidade_futura = _op(cidade="Florianópolis", curso="Na cidade, futura", inicio="2026-08-01", fim="2026-08-20")

        resultado = recomendar(
            "Florianópolis", date(2026, 7, 10), oportunidades=[fora_do_alcance_aberta, na_cidade_futura]
        )

        assert resultado["na_cidade"] == []
        assert resultado["outras_cidades"] == []
        assert resultado["proxima"].curso == "Na cidade, futura"

    def test_alcance_invalido_levanta_erro(self):
        with pytest.raises(ValueError):
            recomendar("Florianópolis", date(2026, 7, 10), alcance="pais_inteiro", oportunidades=[])


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
