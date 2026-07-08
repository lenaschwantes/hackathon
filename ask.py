"""CLI de pergunta do Decifra (demo).

Recupera os trechos e gera a resposta ancorada, citando a fonte.

Uso
---
    python ask.py "quais cursos técnicos têm inscrição aberta em Florianópolis?"
"""

from __future__ import annotations

import sys

from retrieval.generate import answer


def main() -> None:
    if len(sys.argv) < 2:
        print('Uso: python ask.py "sua pergunta aqui"')
        raise SystemExit(1)

    question = " ".join(sys.argv[1:])
    result = answer(question)

    print("\nPergunta:", question)
    print("\nResposta:\n" + result["answer"])
    if result["sources"]:
        print("\nEditais consultados:")
        for src in result["sources"]:
            print("  -", src)


if __name__ == "__main__":
    main()
