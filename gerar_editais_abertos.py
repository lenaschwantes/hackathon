# gerar_editais_abertos.py
import json
from datetime import date, datetime

with open("data/opportunities.json", encoding="utf-8") as f:
    oportunidades = json.load(f)

hoje = date.today().isoformat()


def formatar_br(data_iso: str) -> str:
    return datetime.strptime(data_iso, "%Y-%m-%d").strftime("%d/%m/%Y")


editais = []
for op in oportunidades:
    if op["inscricao_inicio"] <= hoje <= op["inscricao_fim"]:
        editais.append({
            "nome": f"{op['curso']} — {op['campus']}",
            "prazo_inicio": formatar_br(op["inscricao_inicio"]),
            "prazo_fim": formatar_br(op["inscricao_fim"]),
            "link_inscricao": op["link_edital"],
            "forma_ingresso": op["forma_ingresso"],
            "link_pdf": op["link_edital"],
        })

with open("data/editais_abertos.json", "w", encoding="utf-8") as f:
    json.dump(editais, f, ensure_ascii=False, indent=2)

print(f"{len(editais)} editais abertos hoje ({hoje}), de {len(oportunidades)} no total.")