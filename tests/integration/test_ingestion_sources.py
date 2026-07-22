"""Testes das fontes de editais (crawler, pasta local, fallback).

Rodam sem infra: o crawler nunca bate na rede de verdade (`_fetch` é
mockado via monkeypatch), e a fonte local usa `tmp_path`.

    python -m pytest tests/test_ingestion_sources.py -v
"""

from __future__ import annotations

import pytest

from ingestion.sources.base import EditalRef, EditalSource
from ingestion.sources.fallback import FallbackEditalSource
from ingestion.sources.ifsc_crawler import (
    URL_EDITAIS_ABERTOS,
    URL_EDITAIS_ENCERRADOS,
    CrawlerIndisponivel,
    IFSCCrawler,
)
from ingestion.sources.local_folder import LocalFolderSource

HTML_VALIDO = """
<html><body>
<ul>
  <li><a href="/documents/d/ingresso/edital-01-2026-tecnico">Edital 01/2026 - Técnico</a></li>
  <li><a href="/documents/d/ingresso/edital-02-2026-superior">Edital 02/2026 - Superior</a></li>
  <li><a href="/noticias/alguma-coisa">Link qualquer, não é edital</a></li>
  <li><a>Link sem href nenhum</a></li>
</ul>
</body></html>
"""

HTML_COM_ITEM_MALFORMADO = """
<html><body>
<ul>
  <li><a href="/documents/d/ingresso/edital-03-2026-fic">Edital 03/2026 - FIC</a></li>
  <li><a href="http://[invalid/documents/d/ingresso/edital-99.pdf">Edital Malformado</a></li>
</ul>
</body></html>
"""


class TestLocalFolderSource:
    def test_lista_pdfs_da_pasta(self, tmp_path):
        (tmp_path / "edital-a.pdf").write_bytes(b"conteudo a")
        (tmp_path / "edital-b.docx").write_bytes(b"conteudo b")
        (tmp_path / "ignorado.txt").write_text("não é edital")

        refs = LocalFolderSource(str(tmp_path)).list_editais()

        titulos = sorted(r.titulo for r in refs)
        assert titulos == ["edital-a", "edital-b"]
        assert all(r.status == "aberto" for r in refs)

    def test_pasta_inexistente_devolve_lista_vazia(self, tmp_path):
        refs = LocalFolderSource(str(tmp_path / "nao-existe")).list_editais()
        assert refs == []


class TestIFSCCrawler:
    def _crawler(self, respostas: dict[str, str | Exception]) -> IFSCCrawler:
        crawler = IFSCCrawler(pausa_segundos=0)

        def _fetch_falso(url: str) -> str:
            resposta = respostas[url]
            if isinstance(resposta, Exception):
                raise resposta
            return resposta

        crawler._fetch = _fetch_falso  # type: ignore[method-assign]
        return crawler

    def test_extrai_editais_com_status_por_pagina(self):
        crawler = self._crawler(
            {URL_EDITAIS_ABERTOS: HTML_VALIDO, URL_EDITAIS_ENCERRADOS: HTML_VALIDO}
        )

        refs = crawler.list_editais()

        assert len(refs) == 4  # 2 editais válidos x 2 páginas
        abertos = [r for r in refs if r.status == "aberto"]
        encerrados = [r for r in refs if r.status == "encerrado"]
        assert len(abertos) == 2
        assert len(encerrados) == 2
        assert all("edital-" in r.pdf_url for r in refs)

    def test_item_malformado_e_pulado_e_logado(self, caplog):
        crawler = self._crawler(
            {URL_EDITAIS_ABERTOS: HTML_COM_ITEM_MALFORMADO, URL_EDITAIS_ENCERRADOS: HTML_VALIDO}
        )

        with caplog.at_level("ERROR"):
            refs = crawler.list_editais()

        titulos = [r.titulo for r in refs]
        assert "Edital Malformado" not in titulos
        assert any(r.titulo == "Edital 03/2026 - FIC" for r in refs)
        assert "malformado" in caplog.text.lower()

    def test_uma_pagina_falha_outra_continua(self):
        crawler = self._crawler(
            {URL_EDITAIS_ABERTOS: RuntimeError("site fora do ar"), URL_EDITAIS_ENCERRADOS: HTML_VALIDO}
        )

        refs = crawler.list_editais()

        assert len(refs) == 2
        assert all(r.status == "encerrado" for r in refs)

    def test_todas_as_paginas_falham_levanta_crawler_indisponivel(self):
        crawler = self._crawler(
            {
                URL_EDITAIS_ABERTOS: RuntimeError("timeout"),
                URL_EDITAIS_ENCERRADOS: RuntimeError("timeout"),
            }
        )

        with pytest.raises(CrawlerIndisponivel):
            crawler.list_editais()


class _FonteQueFalha(EditalSource):
    def list_editais(self) -> list[EditalRef]:
        raise RuntimeError("fonte primária indisponível")


class _FonteFixa(EditalSource):
    def __init__(self, refs: list[EditalRef]):
        self._refs = refs

    def list_editais(self) -> list[EditalRef]:
        return self._refs


class TestFallbackEditalSource:
    def test_usa_primaria_quando_disponivel(self):
        primaria = _FonteFixa([EditalRef("Edital Primário", "http://x/edital.pdf", "aberto")])
        fallback = _FonteFixa([EditalRef("Edital Fallback", "local/edital.pdf", "aberto")])

        refs = FallbackEditalSource(primaria, fallback).list_editais()

        assert [r.titulo for r in refs] == ["Edital Primário"]

    def test_cai_para_fallback_quando_primaria_falha(self):
        fallback = _FonteFixa([EditalRef("Edital Fallback", "local/edital.pdf", "aberto")])

        refs = FallbackEditalSource(_FonteQueFalha(), fallback).list_editais()

        assert [r.titulo for r in refs] == ["Edital Fallback"]
