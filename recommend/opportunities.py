"""Motor de recomendação: filtro estruturado sobre o catálogo de oportunidades.

Caminho separado do RAG em ``retrieval/``: aqui não há busca semântica, é
filtro por nível e modalidade (rígido) e camada de proximidade geográfica
(preferência, não fronteira), com corte temporal por data de inscrição.
O LLM não decide prazo nem data, só recebe o resultado pronto.
"""

from __future__ import annotations

import json
import logging
import unicodedata
from collections.abc import Iterable
from datetime import date
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, ValidationError

logger = logging.getLogger(__name__)

DATA_PATH = Path(__file__).resolve().parent.parent / "data" / "opportunities.json"

# Aproximação grosseira de região por cidade: sem matriz de distância
# real, agrupa pelas mesorregiões de Santa Catarina onde o IFSC atua.
# Cidade fora deste mapa nunca cai na camada "regiao" (só aparece em
# "na_cidade" / "ead" / "outras_cidades"). Trocar por geodados reais
# (ex.: distância rodoviária entre câmpus) se o catálogo crescer.
_REGIOES: dict[str, str] = {
    "florianopolis": "grande_florianopolis",
    "sao jose": "grande_florianopolis",
    "palhoca": "grande_florianopolis",
    "joinville": "norte",
    "jaragua do sul": "norte",
    "canoinhas": "norte",
    "itajai": "vale_do_itajai",
    "gaspar": "vale_do_itajai",
    "criciuma": "sul",
    "ararangua": "sul",
    "tubarao": "sul",
    "chapeco": "oeste",
    "xanxere": "oeste",
    "sao miguel do oeste": "oeste",
    "lages": "serra",
    "cacador": "serra",
}

_CAMADAS = ("na_cidade", "regiao", "ead", "outras_cidades")

# Quais camadas cada "alcance" aceita. `None` (nao informado) usa
# _CAMADAS_DEFAULT: inclusivo, mas nunca extrapola pra outras cidades
# sem a pessoa pedir "qualquer" explicitamente.
_CAMADAS_POR_ALCANCE: dict[str, tuple[str, ...]] = {
    "local": ("na_cidade",),
    "regional": ("na_cidade", "regiao"),
    "ead": ("na_cidade", "ead"),
    "qualquer": _CAMADAS,
}
_CAMADAS_DEFAULT = ("na_cidade", "regiao", "ead")


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


def _dedup(oportunidades: Iterable[Oportunidade]) -> list[Oportunidade]:
    """Remove duplicatas preservando a ordem (primeira ocorrência vence).

    ``link_edital`` é a chave de identidade: cada edital real tem um
    link único, mesmo que apareça mais de uma vez no catálogo.
    """
    vistos: set[str] = set()
    unicas: list[Oportunidade] = []
    for o in oportunidades:
        if o.link_edital in vistos:
            continue
        vistos.add(o.link_edital)
        unicas.append(o)
    return unicas


def _compativel(oportunidade: Oportunidade, nivel_norm: str | None, modalidade_norm: str | None) -> bool:
    """Verifica nível e modalidade -- filtro rígido, independente de geografia.

    Cidade não elimina mais oportunidade nenhuma daqui: ela só decide em
    que camada de proximidade (`_camada`) a oportunidade cai.
    """
    if modalidade_norm is not None and _normaliza(oportunidade.modalidade) != modalidade_norm:
        return False
    if nivel_norm is not None and _normaliza(oportunidade.nivel) != nivel_norm:
        return False
    return True


def _camada(oportunidade: Oportunidade, cidade_norm: str) -> str:
    """Classifica a oportunidade numa camada de proximidade da pessoa.

    EAD dispensa geografia por completo -- cai em "ead" mesmo que a
    cidade cadastrada da oportunidade coincida com a da pessoa. Entre
    as presenciais: mesma cidade é "na_cidade"; mesma região (via
    `_REGIOES`, aproximação sem matriz de distância) é "regiao"; o
    resto é "outras_cidades".
    """
    if oportunidade.modalidade == "EAD":
        return "ead"

    oportunidade_cidade_norm = _normaliza(oportunidade.cidade)
    if oportunidade_cidade_norm == cidade_norm:
        return "na_cidade"

    regiao_pessoa = _REGIOES.get(cidade_norm)
    if regiao_pessoa is not None and regiao_pessoa == _REGIOES.get(oportunidade_cidade_norm):
        return "regiao"

    return "outras_cidades"


def recomendar(
    cidade: str,
    hoje: date,
    nivel: str | None = None,
    modalidade: str | None = None,
    alcance: Literal["local", "regional", "ead", "qualquer"] | None = None,
    *,
    oportunidades: list[Oportunidade] | None = None,
) -> dict:
    """Agrupa oportunidades por camada de proximidade e classifica por calendário.

    Cidade é preferência, não fronteira: em vez de eliminar quem não mora
    na cidade exata, a oportunidade é colocada numa camada (`na_cidade`,
    `regiao`, `ead`, `outras_cidades`) e o parâmetro `alcance` decide
    quais camadas entram no resultado. Nunca deixa a pessoa sem direção:
    se nada está aberto nas camadas aceitas, busca a próxima a abrir
    dentro delas (nunca fora do alcance pedido).

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
    alcance : {"local", "regional", "ead", "qualquer"}, optional
        Quais camadas geográficas aceitar: ``local`` só a cidade;
        ``regional`` cidade + região; ``ead`` cidade + EAD; ``qualquer``
        aceita qualquer cidade. Default (``None``): ``na_cidade`` +
        ``regiao`` + ``ead`` -- inclusivo, mas só extrapola pra outras
        cidades se a pessoa pedir "qualquer" explicitamente.
    oportunidades : list[Oportunidade], optional
        Catálogo já carregado. Default: lê de ``data/opportunities.json``.

    Returns
    -------
    dict
        ``na_cidade``, ``regiao``, ``ead``, ``outras_cidades``: listas de
        oportunidades com inscrição aberta hoje (bordas inclusivas) na
        respectiva camada -- vazia se a camada não foi aceita por
        `alcance` ou se não há nada aberto nela. ``proxima``: a
        oportunidade de menor ``inscricao_inicio`` futura dentre as
        camadas aceitas, ou ``None`` se não houver nenhuma (nem aberta,
        nem futura) em nenhuma delas.
    """
    if alcance is not None and alcance not in _CAMADAS_POR_ALCANCE:
        raise ValueError(f"alcance inválido: {alcance!r}. Valores aceitos: {tuple(_CAMADAS_POR_ALCANCE)}")
    camadas_aceitas = _CAMADAS_POR_ALCANCE[alcance] if alcance is not None else _CAMADAS_DEFAULT

    catalogo = oportunidades if oportunidades is not None else carregar_oportunidades()

    cidade_norm = _normaliza(cidade)
    nivel_norm = _normaliza(nivel) if nivel is not None else None
    modalidade_norm = _normaliza(modalidade) if modalidade is not None else None

    candidatas = _dedup(o for o in catalogo if _compativel(o, nivel_norm, modalidade_norm))

    por_camada: dict[str, list[Oportunidade]] = {camada: [] for camada in _CAMADAS}
    for o in candidatas:
        camada = _camada(o, cidade_norm)
        if camada in camadas_aceitas:
            por_camada[camada].append(o)

    resultado: dict = {
        camada: [o for o in lista if o.inscricao_inicio <= hoje <= o.inscricao_fim]
        for camada, lista in por_camada.items()
    }

    if any(resultado.values()):
        resultado["proxima"] = None
        return resultado

    # sorted() é estável: em empate de inscricao_inicio, prevalece a ordem
    # em que a oportunidade aparece no catálogo.
    pool_aceito = [o for lista in por_camada.values() for o in lista]
    futuras = sorted(
        (o for o in pool_aceito if o.inscricao_inicio > hoje),
        key=lambda o: o.inscricao_inicio,
    )
    resultado["proxima"] = futuras[0] if futuras else None
    return resultado
