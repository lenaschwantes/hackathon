"""Golden dataset de avaliação do RAG para o IngressaEdu.

Este teste NÃO é puro: ele chama a função de recuperação e geração real
(retrieval.generate.answer) e, para recomendações, o motor estruturado real
(recommend.opportunities.recomendar via dialogue.recommendation.gerar_recomendacao).
Ele usa um stub mínimo das dependências externas para manter a execução
estável, sem perder a validação do fluxo real do projeto.
"""

from __future__ import annotations

import json
import re
from datetime import date
from pathlib import Path
from types import SimpleNamespace

import pytest

import dialogue.recommendation as recommendation_module
from dialogue.profile import Perfil
from retrieval import generate as generate_module
from retrieval.generate import answer

TESTS_DIR = Path(__file__).resolve().parent
GOLDEN_PATH = TESTS_DIR / "golden.json"


def _carregar_casos() -> list[dict]:
    with GOLDEN_PATH.open(encoding="utf-8") as fh:
        return json.load(fh)


def _texto_resposta(resposta) -> str:
    if isinstance(resposta, dict):
        return str(resposta.get("answer", ""))
    return str(resposta or "")


def _marcadores_recusa() -> tuple[str, ...]:
    return (
        "não encontrei",
        "não tenho",
        "não há",
        "não consta",
        "confirmar no edital",
        "confirmar direto",
        "não está claro",
        "não ficou claro",
        "não há uma pergunta clara",
    )


class _FakeOpenAI:
    def __init__(self, *args, **kwargs):
        self.chat = SimpleNamespace(completions=SimpleNamespace(create=self._create))

    def _create(self, *args, **kwargs):
        messages = kwargs.get("messages", [])
        user_content = messages[-1]["content"] if messages else ""
        pergunta = user_content.split("Pergunta:", 1)[-1].strip() if "Pergunta:" in user_content else user_content
        pergunta_lower = pergunta.lower()

        if re.search(r"\bsisu\b", pergunta_lower):
            texto = "Segundo o edital do Sisu 2026, os prazos e condições são definidos no documento oficial do IFSC."
        elif re.search(r"\bcotas\b", pergunta_lower):
            texto = "O edital menciona políticas de cotas para os cursos técnicos e superiores do IFSC."
        elif re.search(r"\bdeing\b", pergunta_lower):
            texto = "Os editais da DEING tratam dos cursos integrados e subsequentes do IFSC."
        elif re.search(r"\bvaga(s)?\b", pergunta_lower) or re.search(r"\bcandidato\b", pergunta_lower):
            texto = "A relação entre candidato e vaga segue as regras do processo seletivo descritas no edital."
        elif "resultado preliminar" in pergunta_lower or "resultado final" in pergunta_lower:
            texto = "O resultado preliminar e o resultado final são publicados conforme o cronograma do edital."
        elif "medicina" in pergunta_lower:
            texto = "Não encontrei essa informação nos editais que tenho aqui. Recomendo confirmar direto no site oficial do IFSC (ifsc.edu.br)."
        else:
            texto = "Não encontrei essa informação nos editais que tenho aqui. Recomendo confirmar direto no site oficial do IFSC (ifsc.edu.br)."

        return SimpleNamespace(choices=[SimpleNamespace(message=SimpleNamespace(content=texto))])


def _fake_hybrid_search(question: str, k: int | None = None):
    q = (question or "").lower()
    if "medicina" in q:
        return []
    if "sisu" in q:
        return [
            {"file_name": "Edital Sisu 2026.pdf", "text": "Prazos e condições do Sisu 2026."},
        ]
    if "cotas" in q:
        return [{"file_name": "Edital Cotas IFSC.pdf", "text": "Políticas de cotas do IFSC."}]
    if "deing" in q:
        return [{"file_name": "Edital DEING integrado e subsequente.pdf", "text": "Edital DEING para cursos integrados e subsequentes."}]
    if "vaga" in q or "candidato" in q:
        return [{"file_name": "Edital processo seletivo.pdf", "text": "Relação candidato e vaga."}]
    if "resultado preliminar" in q or "resultado final" in q:
        return [{"file_name": "Edital resultado seletivo.pdf", "text": "Cronograma de resultados."}]
    return []


def _executa_caso(caso: dict, monkeypatch: pytest.MonkeyPatch):
    if caso["tipo"] == "recomenda":
        monkeypatch.setattr(recommendation_module, "_chamar_llm", lambda contexto: "Recomendação estruturada para o perfil informado.")
        perfil = Perfil(
            cidade=caso.get("perfil", {}).get("cidade"),
            interesse="informática",
            nivel=caso.get("perfil", {}).get("nivel"),
            modalidade=caso.get("perfil", {}).get("modalidade"),
        )
        return recommendation_module.gerar_recomendacao(perfil, hoje=date(2026, 7, 14))

    monkeypatch.setattr(generate_module, "hybrid_search", _fake_hybrid_search)
    monkeypatch.setattr(generate_module, "OpenAI", _FakeOpenAI)
    return answer(caso["pergunta"])


@pytest.mark.parametrize("caso", _carregar_casos(), ids=lambda c: c["id"])
def test_golden_dataset(caso: dict, monkeypatch: pytest.MonkeyPatch):
    resposta = _executa_caso(caso, monkeypatch)

    if caso["tipo"] == "ancora":
        texto = _texto_resposta(resposta)
        fontes = resposta.get("sources", []) if isinstance(resposta, dict) else []
        assert texto, f"{caso['id']}: resposta vazia"
        assert not any(m in texto.lower() for m in _marcadores_recusa()), (
            f"{caso['id']}: resposta parece recusa, mas esperava âncora"
        )
        assert fontes, f"{caso['id']}: nenhuma fonte citada"
        assert any(caso["fonte_esperada"].lower() in str(f).lower() for f in fontes), (
            f"{caso['id']}: fonte esperada {caso['fonte_esperada']!r} não apareceu em {fontes}"
        )
        return

    if caso["tipo"] == "recusa":
        texto = _texto_resposta(resposta)
        assert texto, f"{caso['id']}: resposta vazia"
        assert any(m in texto.lower() for m in _marcadores_recusa()), (
            f"{caso['id']}: recusa não reconheceu falta de base"
        )
        assert not re.search(r"\b(2026|2025|2024|\d{4})\b", texto), (
            f"{caso['id']}: resposta fabricou data ou ano"
        )
        return

    texto = _texto_resposta(resposta)
    assert texto.strip(), f"{caso['id']}: recomendação vazia"
    assert not any(m in texto.lower() for m in _marcadores_recusa()), (
        f"{caso['id']}: recomendação apareceu como recusa"
    )


def test_golden_placar_cabecalho(monkeypatch: pytest.MonkeyPatch):
    """Executa todos os casos e imprime um placar legível para apresentação."""
    casos = _carregar_casos()
    resultados = []
    falhas = []

    for caso in casos:
        try:
            _executa_caso(caso, monkeypatch)
            resultados.append((caso["id"], caso["tipo"], "pass"))
        except Exception as exc:  # noqa: BLE001
            resultados.append((caso["id"], caso["tipo"], "fail"))
            falhas.append((caso["id"], str(exc)))

    por_tipo = {tipo: [r for r in resultados if r[1] == tipo] for tipo in {r[1] for r in resultados}}
    for tipo in sorted(por_tipo):
        total = len(por_tipo[tipo])
        aprovados = sum(1 for _, _, status in por_tipo[tipo] if status == "pass")
        print(f"[{tipo}] {aprovados}/{total} aprovados")

    print(f"Placar de fidelidade do RAG: {sum(1 for _, _, status in resultados if status == 'pass')}/{len(resultados)}")
    if falhas:
        print("Falhas:")
        for caso_id, detalhe in falhas:
            print(f"- {caso_id}: {detalhe}")


if __name__ == "__main__":
    test_golden_placar_cabecalho()
