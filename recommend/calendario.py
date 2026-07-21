"""Calendário de inscrições do IFSC: segunda fonte estruturada da recomendação.

Consultado só quando o catálogo de oportunidades concretas
(``recommend/opportunities.py``) não tem nada aberto nem uma próxima vaga
específica pro nível da pessoa -- ver ``dialogue/recommendation.py`` pra
ordem de precedência. Puro, sem LLM: datas nunca passam pelo modelo, só a
redação final da mensagem.
"""

from __future__ import annotations

import json
import logging
import unicodedata
from datetime import date
from pathlib import Path
from typing import Literal, Optional

from pydantic import BaseModel, ValidationError, model_validator

logger = logging.getLogger(__name__)

DATA_PATH = Path(__file__).resolve().parent.parent / "data" / "calendario.json"

_NIVEIS = (
    "técnico integrado",
    "técnico subsequente",
    "proeja",
    "superior",
    "especialização",
    "mestrado",
    "FIC",
)

_FORMAS_INGRESSO = (
    "cadastro de reserva",
    "inscrição",
    "prova",
    "sorteio",
    "ordem de inscrição",
    "vestibular",
    "Sisu",
)


class Janela(BaseModel):
    """Uma janela de inscrição do calendário oficial do IFSC, por nível/tipo de curso.

    ``inicio``/``fim`` são ``None`` quando ``data_confirmada`` é ``False`` --
    o IFSC já anunciou que vai ter aquela janela, mas ainda não publicou
    data (ex.: Sisu "conforme cronograma do MEC"). Nesse caso a janela nunca
    aparece em ``abertas_agora`` nem em ``proxima`` (não há data pra
    comparar/ordenar): ela só é reportada como "existe, mas sem data ainda"
    em ``a_confirmar``, pra nunca inventar uma data falsa.
    """

    nivel: Literal[_NIVEIS]
    forma_ingresso: Literal[_FORMAS_INGRESSO]
    semestre_letivo: str
    inicio: Optional[date] = None
    fim: Optional[date] = None
    data_confirmada: bool = True
    observacao: Optional[str] = None

    @model_validator(mode="after")
    def _valida_datas(self) -> "Janela":
        if self.data_confirmada:
            if self.inicio is None or self.fim is None:
                raise ValueError("janela com data_confirmada=True precisa de inicio e fim")
            if self.fim < self.inicio:
                raise ValueError("fim anterior ao inicio")
        elif self.inicio is not None or self.fim is not None:
            raise ValueError("janela com data_confirmada=False não deve ter inicio/fim preenchido")
        return self


def carregar_calendario(path: Path | str = DATA_PATH) -> list[Janela]:
    """Carrega e valida o catálogo de janelas de inscrição.

    Registro malformado (data inválida, campo ausente, valor fora do
    vocabulário controlado, ou inconsistência entre ``data_confirmada`` e
    ``inicio``/``fim``) é descartado com log, sem derrubar a carga dos
    demais.

    Parameters
    ----------
    path : Path or str, optional
        Caminho do JSON. Default: ``data/calendario.json``.

    Returns
    -------
    list[Janela]
        Registros válidos do catálogo.
    """
    with open(path, encoding="utf-8") as f:
        bruto = json.load(f)

    janelas = []
    for i, registro in enumerate(bruto):
        try:
            janelas.append(Janela(**registro))
        except ValidationError as exc:
            logger.warning("Registro %d de %s descartado: %s", i, path, exc)
    return janelas


def _normaliza(texto: str) -> str:
    """Normaliza texto para comparação: minúsculo e sem acento."""
    sem_acento = unicodedata.normalize("NFKD", texto).encode("ascii", "ignore").decode()
    return sem_acento.strip().lower()


def consultar_calendario(
    nivel: str | None,
    hoje: date,
    *,
    janelas: list[Janela] | None = None,
) -> dict:
    """Consulta as janelas de inscrição do calendário pro nível da pessoa.

    Parameters
    ----------
    nivel : str, optional
        Nível de curso desejado (ex.: "técnico integrado", "superior").
        Sem nível, devolve um resultado vazio -- não há o que consultar.
    hoje : date
        Data de referência para o corte temporal, injetada pelo chamador
        (nunca calculada internamente), igual ao contrato de
        ``recommend.opportunities.recomendar``.
    janelas : list[Janela], optional
        Catálogo já carregado. Default: lê de ``data/calendario.json``.

    Returns
    -------
    dict
        ``abertas_agora``: janelas com data confirmada cujo período cobre
        hoje (bordas inclusivas). ``proxima``: a janela de menor ``inicio``
        futura dentre as de data confirmada, ou ``None`` se
        ``abertas_agora`` não for vazia ou se não houver nenhuma janela
        futura. ``a_confirmar``: janelas do nível sem data ainda publicada
        (``data_confirmada=False``) -- nunca entram em ``abertas_agora``
        nem em ``proxima``, só aqui, pra nunca virar uma data inventada.
    """
    if not nivel:
        return {"abertas_agora": [], "proxima": None, "a_confirmar": []}

    catalogo = janelas if janelas is not None else carregar_calendario()
    nivel_norm = _normaliza(nivel)

    do_nivel = [j for j in catalogo if _normaliza(j.nivel) == nivel_norm]
    com_data = [j for j in do_nivel if j.data_confirmada]
    a_confirmar = [j for j in do_nivel if not j.data_confirmada]

    abertas_agora = [j for j in com_data if j.inicio <= hoje <= j.fim]

    if abertas_agora:
        proxima = None
    else:
        # sorted() é estável: em empate de inicio, prevalece a ordem em
        # que a janela aparece no catálogo.
        futuras = sorted((j for j in com_data if j.inicio > hoje), key=lambda j: j.inicio)
        proxima = futuras[0] if futuras else None

    return {"abertas_agora": abertas_agora, "proxima": proxima, "a_confirmar": a_confirmar}
