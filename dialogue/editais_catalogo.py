"""
Catálogo manual de editais abertos, usado pelo fluxo "Dúvidas sobre
prazos e formas de ingresso" do menu inicial (RAG manual). Precisa ser
revisado periodicamente pra remover editais já encerrados.
"""
import json
from pathlib import Path

_CATALOGO_PATH = Path(__file__).parent.parent / "data" / "editais_abertos.json"


def carregar_editais_abertos() -> list[dict]:
    """Lê o catálogo do disco a cada chamada -- lista pequena, não vale
    cachear e arriscar servir dado desatualizado após edição manual."""
    with open(_CATALOGO_PATH, encoding="utf-8") as f:
        return json.load(f)


def buscar_edital_por_indice(indice: int) -> dict | None:
    editais = carregar_editais_abertos()
    if 0 <= indice < len(editais):
        return editais[indice]
    return None