"""
Fonte que descobre editais nas páginas públicas de listagem do IFSC.

Frágil por natureza — depende da estrutura HTML atual do site, que pode
mudar sem aviso. Por isso fica isolado atrás de `EditalSource`: quando o
HTML do IFSC mudar, só este arquivo muda, nada mais no pipeline.

O status (aberto/encerrado) vem de qual página listou o edital, nunca é
inferido por data.
"""

from __future__ import annotations

import logging
import time
from urllib.parse import urljoin

import httpx
from bs4 import BeautifulSoup

from ingestion.sources.base import EditalRef, EditalSource, Status

logger = logging.getLogger(__name__)

URL_EDITAIS_ABERTOS = "https://www.ifsc.edu.br/editais-com-inscricoes-abertas"
URL_EDITAIS_ENCERRADOS = "https://www.ifsc.edu.br/editais-com-inscricoes-encerradas"

_PADRAO_PDF = "/documents/d/ingresso/edital-"
_USER_AGENT = (
    "DecifraBot/1.0 (+https://github.com/lenaschwantes/hackathon; "
    "ingestao automatica de editais publicos do IFSC)"
)
_TIMEOUT_SEGUNDOS = 15.0
_PAUSA_ENTRE_PAGINAS_SEGUNDOS = 1.0


class CrawlerIndisponivel(RuntimeError):
    """Levantado quando nenhuma das páginas de listagem pôde ser acessada."""


class IFSCCrawler(EditalSource):
    """Lê `editais-com-inscricoes-abertas` e `-encerradas` e extrai os PDFs."""

    def __init__(
        self,
        paginas: dict[str, Status] | None = None,
        pausa_segundos: float = _PAUSA_ENTRE_PAGINAS_SEGUNDOS,
    ):
        self._paginas = paginas or {
            URL_EDITAIS_ABERTOS: "aberto",
            URL_EDITAIS_ENCERRADOS: "encerrado",
        }
        self._pausa_segundos = pausa_segundos

    def list_editais(self) -> list[EditalRef]:
        """Busca as páginas de listagem e extrai os editais encontrados.

        Falha ao buscar uma página é logada e a outra ainda é tentada.
        Só levanta `CrawlerIndisponivel` se NENHUMA página respondeu —
        sinal pro `FallbackEditalSource` cair pra pasta local.
        """
        refs: list[EditalRef] = []
        paginas_ok = 0

        for i, (url, status) in enumerate(self._paginas.items()):
            if i > 0 and self._pausa_segundos:
                time.sleep(self._pausa_segundos)
            try:
                html = self._fetch(url)
            except Exception as exc:
                logger.error("Falha ao buscar página de editais: %s (%s)", url, type(exc).__name__)
                continue
            paginas_ok += 1
            refs.extend(self._parse_pagina(html, status, base_url=url))

        if paginas_ok == 0:
            raise CrawlerIndisponivel(
                "Não foi possível acessar nenhuma página de editais do IFSC."
            )
        return refs

    def _fetch(self, url: str) -> str:
        resp = httpx.get(
            url,
            timeout=_TIMEOUT_SEGUNDOS,
            headers={"User-Agent": _USER_AGENT},
            follow_redirects=True,
        )
        resp.raise_for_status()
        return resp.text

    def _parse_pagina(self, html: str, status: Status, base_url: str) -> list[EditalRef]:
        refs: list[EditalRef] = []
        soup = BeautifulSoup(html, "html.parser")

        for link in soup.find_all("a"):
            try:
                ref = self._parse_link(link, status, base_url)
            except Exception as exc:
                logger.error("Item de edital malformado, pulando (%s)", type(exc).__name__)
                continue
            if ref is not None:
                refs.append(ref)

        return refs

    def _parse_link(self, link, status: Status, base_url: str) -> EditalRef | None:
        href = link.get("href")
        if not href:
            return None

        pdf_url = urljoin(base_url, href)
        if _PADRAO_PDF not in pdf_url:
            return None

        titulo = link.get_text(strip=True) or pdf_url.rsplit("/", 1)[-1]
        return EditalRef(titulo=titulo, pdf_url=pdf_url, status=status)
