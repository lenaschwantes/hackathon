"""Testes da detecção incremental de editais novos.

Puro: usa uma fonte fake e um callable fake pra "já conhecido", sem
Weaviate. Confirma que rodar a mesma fonte de novo não reprocessa.

    python -m pytest tests/test_ingestion_discovery.py -v
"""

from __future__ import annotations

from ingestion.discovery import descobrir_novos
from ingestion.sources.base import EditalRef, EditalSource

REF_A = EditalRef("Edital A", "http://x/edital-a.pdf", "aberto")
REF_B = EditalRef("Edital B", "http://x/edital-b.pdf", "encerrado")


class _FonteFixa(EditalSource):
    def __init__(self, refs: list[EditalRef]):
        self._refs = refs

    def list_editais(self) -> list[EditalRef]:
        return self._refs


class TestDescobrirNovos:
    def test_primeira_vez_devolve_tudo(self):
        source = _FonteFixa([REF_A, REF_B])

        novos = descobrir_novos(source, ja_conhecido=lambda url: False)

        assert novos == [REF_A, REF_B]

    def test_filtra_os_ja_conhecidos(self):
        source = _FonteFixa([REF_A, REF_B])
        conhecidos = {REF_A.pdf_url}

        novos = descobrir_novos(source, ja_conhecido=conhecidos.__contains__)

        assert novos == [REF_B]

    def test_segundo_run_apos_ingestao_nao_reprocessa(self):
        source = _FonteFixa([REF_A, REF_B])
        processados: set[str] = set()

        primeiro_run = descobrir_novos(source, ja_conhecido=processados.__contains__)
        assert primeiro_run == [REF_A, REF_B]

        for ref in primeiro_run:
            processados.add(ref.pdf_url)

        segundo_run = descobrir_novos(source, ja_conhecido=processados.__contains__)
        assert segundo_run == []
