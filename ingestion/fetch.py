"""
Baixa o conteúdo de um `EditalRef.pdf_url`, seja ele uma URL http(s) ou um
caminho de arquivo local. Fica em memória — nada é gravado em disco.
"""

from __future__ import annotations

from pathlib import Path

import httpx

_TIMEOUT_SEGUNDOS = 30.0
_USER_AGENT = (
    "DecifraBot/1.0 (+https://github.com/lenaschwantes/hackathon; "
    "ingestao automatica de editais publicos do IFSC)"
)


def baixar_conteudo(pdf_url: str) -> bytes:
    if pdf_url.startswith("http://") or pdf_url.startswith("https://"):
        resp = httpx.get(
            pdf_url,
            timeout=_TIMEOUT_SEGUNDOS,
            headers={"User-Agent": _USER_AGENT},
            follow_redirects=True,
        )
        resp.raise_for_status()
        return resp.content

    return Path(pdf_url).read_bytes()
