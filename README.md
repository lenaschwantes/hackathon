# Decifra

Assistente conversacional que traduz editais da educação pública para linguagem simples, ancorado em RAG. Entregue via Telegram, orienta o cidadão sobre cursos e formas de ingresso no IFSC.

Projeto da 1ª Jornada Incubintech, Inovação Aberta, Desafio 12 (LabCiDig).

## Arquitetura

O perfil do cidadão alimenta dois motores separados, e eles não se misturam:

- **Recomendação (estruturado):** qual curso, câmpus, inscrição aberta ou não. Sai de uma consulta com filtro por localização e calendário. O LLM não decide data, recebe o resultado pronto.
- **Tradução (RAG):** o que significa a cota, requisitos, forma de ingresso. Sai da recuperação híbrida ancorada no edital, sempre citando a fonte.

Essa separação é o que garante fidelidade: o assistente nunca inventa prazo, e quando não há base no acervo, reconhece que não sabe em vez de chutar.

## Estrutura

```
config/            settings central (Weaviate, Voyage, chunking, retrieval)
ingestion/         extração, limpeza, chunking, embedding e store no Weaviate
retrieval/         busca híbrida (vetor + BM25)
utils/             hashing e validação
data/editais/      PDFs dos editais do IFSC (não versionados)
run_ingest.py      CLI: percorre os editais e indexa
docker-compose.yml weaviate + rabbitmq + redis
```

## Como rodar

```bash
cp .env.example .env      # preencha VOYAGE_API_KEY e ANTHROPIC_API_KEY
docker compose up -d weaviate rabbitmq redis
uv sync                   # ou pip install -e .
# coloque os PDFs em data/editais/
python run_ingest.py
```

## Reaproveitamento

O core de RAG vem de uma plataforma de produção. Os módulos de extração, limpeza, chunking, embedding e o store idempotente do Weaviate são reaproveitados; a fonte de documentos foi trocada de storage remoto para leitura de arquivo local, e a camada de canal e recomendação é nova.

## Próximos passos (fases da PoC)

- [ ] Fase 0: portar chunk + embed no `pipeline.py` (marcado com TODO) e validar o retrieval no primeiro edital
- [ ] Fase 1: tabela estruturada de oportunidades (`data/opportunities.json`) + filtro geo e temporal
- [ ] Fase 2: diálogo de perfil (structured output) + prompt de redação acessível
- [ ] Fase 3: canal do Telegram + sessão no Redis
- [ ] Fase 4: orquestração dos dois motores + testes de recusa
