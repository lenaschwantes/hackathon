"""Motor de recomendação: filtro estruturado sobre o catálogo de oportunidades.

Caminho separado do RAG em ``retrieval/``: aqui não há busca semântica, é
filtro por cidade, nível e modalidade, com corte temporal por data de
inscrição. O LLM não decide prazo nem data, só recebe o resultado pronto.
"""

from __future__ import annotations

import json
import logging
import unicodedata
from datetime import date
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, ValidationError

logger = logging.getLogger(__name__)

DATA_PATH = Path(__file__).resolve().parent.parent / "data" / "opportunities.json"


class Oportunidade(BaseModel):
    """Uma vaga de curso no catálogo estruturado de oportunidades do IFSC."""

    curso: str
    campus: str
    cidade: str
    modalidade: Literal["presencial", "EAD"]
    nivel: Literal["técnico integrado", "técnico subsequente", "superior", "FIC"]
    forma_ingresso: Literal["sorteio", "prova", "ENEM", "análise"]
    inscricao_inicio: date
    inscricao_fim: date
    link_edital: str


def carregar_oportunidades(path: Path | str = DATA_PATH) -> list[Oportunidade]:
    """Carrega e valida o catálogo de oportunidades.

    Registro malformado (data inválida, campo ausente, valor fora do
    vocabulário controlado) é descartado com log, sem derrubar a carga
    dos demais.

    Parameters
    ----------
    path : Path or str, optional
        Caminho do JSON. Default: ``data/opportunities.json``.

    Returns
    -------
    list[Oportunidade]
        Registros válidos do catálogo.
    """
    with open(path, encoding="utf-8") as f:
        bruto = json.load(f)

    oportunidades = []
    for i, registro in enumerate(bruto):
        try:
            oportunidades.append(Oportunidade(**registro))
        except ValidationError as exc:
            logger.warning("Registro %d de %s descartado: %s", i, path, exc)
    return oportunidades


def _normaliza(texto: str) -> str:
    """Normaliza texto para comparação: minúsculo e sem acento."""
    sem_acento = unicodedata.normalize("NFKD", texto).encode("ascii", "ignore").decode()
    return sem_acento.strip().lower()


def _compativel(
    oportunidade: Oportunidade,
    cidade_norm: str,
    nivel_norm: str | None,
    modalidade_norm: str | None,
) -> bool:
    """Verifica se a oportunidade atende ao perfil pedido.

    EAD dispensa o filtro geográfico: uma vez que modalidade e nível batem,
    a oportunidade é compatível independente da cidade do cidadão.
    """
    if modalidade_norm is not None and _normaliza(oportunidade.modalidade) != modalidade_norm:
        return False
    if nivel_norm is not None and _normaliza(oportunidade.nivel) != nivel_norm:
        return False
    if oportunidade.modalidade == "EAD":
        return True
    return _normaliza(oportunidade.cidade) == cidade_norm


def recomendar(
    cidade: str,
    hoje: date,
    nivel: str | None = None,
    modalidade: str | None = None,
    *,
    oportunidades: list[Oportunidade] | None = None,
) -> dict:
    """Filtra oportunidades por perfil e classifica por calendário.

    Nunca retorna as duas listas vazias sem motivo: se não há inscrição
    aberta compatível, busca a próxima a abrir. O cidadão sempre sai com
    uma direção, mesmo que seja "ainda não abriu".

    Parameters
    ----------
    cidade : str
        Cidade do cidadão. Comparação insensível a caixa e acento.
    hoje : date
        Data de referência para o corte temporal, injetada pelo chamador
        (nunca calculada internamente), para a função ser determinística
        e testável.
    nivel : str, optional
        Nível de ensino desejado (ex.: "técnico integrado", "superior").
    modalidade : str, optional
        "presencial" ou "EAD".
    oportunidades : list[Oportunidade], optional
        Catálogo já carregado. Default: lê de ``data/opportunities.json``.

    Returns
    -------
    dict
        ``abertas``: oportunidades com inscrição aberta hoje (bordas
        inclusivas), compatíveis com o perfil. ``proxima``: a oportunidade
        compatível de menor ``inscricao_inicio`` futura, ou ``None`` se não
        houver nenhuma compatível (nem aberta, nem futura).
    """
    catalogo = oportunidades if oportunidades is not None else carregar_oportunidades()

    cidade_norm = _normaliza(cidade)
    nivel_norm = _normaliza(nivel) if nivel is not None else None
    modalidade_norm = _normaliza(modalidade) if modalidade is not None else None

    candidatas = [
        o for o in catalogo if _compativel(o, cidade_norm, nivel_norm, modalidade_norm)
    ]

    abertas = [o for o in candidatas if o.inscricao_inicio <= hoje <= o.inscricao_fim]
    if abertas:
        return {"abertas": abertas, "proxima": None}

    # sorted() é estável: em empate de inscricao_inicio, prevalece a ordem
    # em que a oportunidade aparece no catálogo.
    futuras = sorted(
        (o for o in candidatas if o.inscricao_inicio > hoje),
        key=lambda o: o.inscricao_inicio,
    )
    proxima = futuras[0] if futuras else None
    return {"abertas": [], "proxima": proxima}
