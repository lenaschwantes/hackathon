"""Testes de ingestão isolada por edital: retry, dead-letter e isolamento.

Puro: mocka `baixar_conteudo` e `ingest_document`, nunca bate em rede nem
em Weaviate. `sleep_fn` é substituído por um no-op pra não esperar de
verdade o backoff exponencial.

    python -m pytest tests/test_ingestion_auto_ingest.py -v
"""

from __future__ import annotations

import ingestion.auto_ingest as auto_ingest_module
from ingestion.auto_ingest import ingerir_edital
from ingestion.sources.base import EditalRef

REF = EditalRef("Edital Teste", "http://x/edital-teste.pdf", "aberto")


def _sem_espera(_segundos: float) -> None:
    return None


class TestRetry:
    def test_sucesso_de_primeira_nao_tenta_de_novo(self, monkeypatch):
        chamadas = {"baixar": 0}

        def baixar_ok(_url: str) -> bytes:
            chamadas["baixar"] += 1
            return b"conteudo"

        monkeypatch.setattr(auto_ingest_module, "baixar_conteudo", baixar_ok)
        monkeypatch.setattr(
            auto_ingest_module, "ingest_document", lambda *a, **k: {"status": "indexed"}
        )

        resultado = ingerir_edital(REF, store=object(), sleep_fn=_sem_espera)

        assert resultado["status"] == "indexed"
        assert chamadas["baixar"] == 1

    def test_falha_duas_vezes_sucesso_na_terceira(self, monkeypatch):
        tentativas = {"n": 0}

        def baixar_com_falhas(_url: str) -> bytes:
            tentativas["n"] += 1
            if tentativas["n"] < 3:
                raise ConnectionError("timeout simulado")
            return b"conteudo"

        monkeypatch.setattr(auto_ingest_module, "baixar_conteudo", baixar_com_falhas)
        monkeypatch.setattr(
            auto_ingest_module, "ingest_document", lambda *a, **k: {"status": "indexed"}
        )

        resultado = ingerir_edital(REF, store=object(), sleep_fn=_sem_espera)

        assert resultado["status"] == "indexed"
        assert tentativas["n"] == 3

    def test_esgota_tentativas_devolve_failed_sem_levantar(self, monkeypatch, caplog):
        def baixar_sempre_falha(_url: str) -> bytes:
            raise ConnectionError("site fora do ar")

        monkeypatch.setattr(auto_ingest_module, "baixar_conteudo", baixar_sempre_falha)

        with caplog.at_level("ERROR"):
            resultado = ingerir_edital(REF, store=object(), sleep_fn=_sem_espera)

        assert resultado["status"] == "failed"
        assert "DEAD-LETTER" in caplog.text


class TestIsolamentoEntreEditais:
    def test_falha_em_um_nao_impede_o_proximo(self, monkeypatch):
        ref_falha = EditalRef("Edital Quebrado", "http://x/quebrado.pdf", "aberto")
        ref_ok = EditalRef("Edital OK", "http://x/ok.pdf", "aberto")

        def baixar(url: str) -> bytes:
            if "quebrado" in url:
                raise ConnectionError("PDF corrompido")
            return b"conteudo"

        monkeypatch.setattr(auto_ingest_module, "baixar_conteudo", baixar)
        monkeypatch.setattr(
            auto_ingest_module, "ingest_document", lambda *a, **k: {"status": "indexed"}
        )

        resultado_falha = ingerir_edital(ref_falha, store=object(), sleep_fn=_sem_espera)
        resultado_ok = ingerir_edital(ref_ok, store=object(), sleep_fn=_sem_espera)

        assert resultado_falha["status"] == "failed"
        assert resultado_ok["status"] == "indexed"
