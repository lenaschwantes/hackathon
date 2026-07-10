"""
Composição de fontes: tenta a primária, cai pra secundária se ela falhar.

É isso que garante que a pasta local continua funcionando como fallback
manual — se o site do IFSC mudar de estrutura ou ficar fora do ar
(`CrawlerIndisponivel` ou qualquer outra exceção), o pipeline não trava.
"""

from __future__ import annotations

import logging

from ingestion.sources.base import EditalRef, EditalSource

logger = logging.getLogger(__name__)


class FallbackEditalSource(EditalSource):
    def __init__(self, primaria: EditalSource, fallback: EditalSource):
        self._primaria = primaria
        self._fallback = fallback

    def list_editais(self) -> list[EditalRef]:
        try:
            refs = self._primaria.list_editais()
            logger.info(
                "Fonte primária (%s) OK: %d editais", type(self._primaria).__name__, len(refs)
            )
            return refs
        except Exception:
            logger.exception(
                "Fonte primária (%s) falhou, caindo para fallback (%s)",
                type(self._primaria).__name__,
                type(self._fallback).__name__,
            )
            return self._fallback.list_editais()
