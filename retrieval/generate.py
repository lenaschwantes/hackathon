"""Geração ancorada da resposta.

Junta os dois lados do RAG: recupera os trechos e pede ao LLM que responda
SOMENTE com base neles, citando o edital de origem. Se não houver base,
reconhece que não sabe em vez de inventar. É aqui que mora o critério de
fidelidade da banca.
"""

from __future__ import annotations

import time

import openai
from openai import OpenAI

from config.settings import settings
from retrieval.search import hybrid_search

SYSTEM = (
    "Você é um orientador que traduz editais do IFSC em linguagem simples e "
    "acolhedora, para pessoas com diferentes graus de letramento. "
    "Responda SOMENTE com base nos trechos fornecidos. "
    "Se a resposta não estiver nos trechos, diga com clareza que não encontrou "
    "essa informação no acervo e oriente a pessoa a confirmar no edital oficial "
    "do IFSC. NUNCA invente prazo, requisito, curso ou modalidade. "
    "Sempre indique de qual edital veio a informação."
)

_SEM_BASE = (
    "Não encontrei essa informação nos editais que tenho aqui. "
    "Recomendo confirmar direto no site oficial do IFSC (ifsc.edu.br)."
)


def answer(question: str, k: int | None = None) -> dict:
    """Responde a pergunta ancorada nos editais.

    Parameters
    ----------
    question : str
        Pergunta do cidadão.
    k : int, optional
        Número de trechos a recuperar.

    Returns
    -------
    dict
        ``answer`` (texto) e ``sources`` (lista de editais citados).
    """
    hits = hybrid_search(question, k=k)
    if not hits:
        return {"answer": _SEM_BASE, "sources": []}

    contexto = "\n\n".join(
        f"[Fonte: {h['file_name']}]\n{h['text']}" for h in hits
    )
    client = OpenAI(api_key=settings.groq_api_key, base_url="https://api.groq.com/openai/v1")

    delays = (2, 4, 8)
    for attempt, delay in enumerate((0, *delays)):
        if delay:
            time.sleep(delay)
        try:
            msg = client.chat.completions.create(
                model=settings.groq_model,
                max_tokens=1000,
                messages=[
                    {"role": "system", "content": SYSTEM},
                    {
                        "role": "user",
                        "content": f"Trechos dos editais:\n{contexto}\n\nPergunta: {question}",
                    },
                ],
            )
            break
        except openai.RateLimitError:
            if attempt == len(delays):
                raise
            continue

    text = msg.choices[0].message.content
    sources = sorted({h["file_name"] for h in hits if h["file_name"]})
    return {"answer": text, "sources": sources}
