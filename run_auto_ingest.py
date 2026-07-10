"""CLI de ingestão automática de editais do IFSC.

Descobre editais novos no site do IFSC (com fallback pra pasta local se o
site estiver indisponível) e ingere cada um.

Uso
---
    python run_auto_ingest.py            # roda um ciclo e termina
    python run_auto_ingest.py --loop     # roda em loop, a cada
                                          # settings.auto_ingest_ciclo_segundos
"""

from __future__ import annotations

from ingestion.runner import main

if __name__ == "__main__":
    main()
