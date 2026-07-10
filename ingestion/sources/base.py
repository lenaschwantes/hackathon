"""
Interface mínima que toda fonte de editais precisa seguir.

Uma "fonte" descobre *onde* estão os editais (site do IFSC, pasta local,
etc.) e devolve só metadados — nunca baixa nem processa o conteúdo do PDF.
Isso é o que isola o pipeline de ingestão (robusto) da forma como cada
fonte descobre documentos (frágil, principalmente no caso do crawler HTML).
Quando o site do IFSC mudar de estrutura, só `ifsc_crawler.py` muda.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Literal

Status = Literal["aberto", "encerrado"]


@dataclass(frozen=True)
class EditalRef:
    """Referência a um edital, antes de qualquer download.

    Attributes
    ----------
    titulo : str
        Título ou nome do edital, como aparece na fonte.
    pdf_url : str
        Onde baixar o PDF — URL http(s) ou caminho de arquivo local.
    status : Status
        "aberto" ou "encerrado", conforme a fonte de origem informa.
        Nunca inferido por data.
    """

    titulo: str
    pdf_url: str
    status: Status


class EditalSource(ABC):
    """Contrato que toda fonte de editais tem que implementar."""

    @abstractmethod
    def list_editais(self) -> list[EditalRef]:
        """Lista os editais disponíveis nesta fonte, sem baixar o conteúdo."""
        raise NotImplementedError
