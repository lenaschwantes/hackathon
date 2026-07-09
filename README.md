# Decifra

Assistente conversacional que traduz editais da educação pública para linguagem simples, ancorado em RAG. Entregue via Telegram, orienta o cidadão sobre cursos e formas de ingresso no IFSC.

Projeto da 1ª Jornada Incubintech, Inovação Aberta, Desafio 12 (LabCiDig).

## Arquitetura

O perfil do cidadão alimenta dois motores separados, e eles não se misturam:

- **Recomendação (estruturado):** qual curso, câmpus, inscrição aberta ou não. Sai de uma consulta com filtro por localização e calendário. O LLM não decide data, recebe o resultado pronto.
- **Tradução (RAG):** o que significa a cota, requisitos, forma de ingresso. Sai da recuperação híbrida ancorada no edital, sempre citando a fonte.

Essa separação é o que garante fidelidade: o assistente nunca inventa prazo, e quando não há base no acervo, reconhece que não sabe em vez de chutar.

Stack: embeddings com Voyage (`voyage-3`), geração com Groq (`llama-3.3-70b-versatile`, API compatível com OpenAI), store vetorial no Weaviate com busca híbrida (vetor + BM25).

## Estrutura

```
config/            settings central (Weaviate, Voyage, Groq, chunking, retrieval)
ingestion/         extração, limpeza, chunking, embedding e store no Weaviate
retrieval/         busca híbrida (vetor + BM25) + geração ancorada via Groq
recommend/         motor estruturado: filtro de oportunidades por perfil
channels/          adaptadores de canal (Telegram) + sessão no Redis + engine.py (liga o canal ao RAG)
utils/             hashing e validação
tests/             testes puros (rodam sem infra e sem chaves de API)
data/editais/      PDFs dos editais do IFSC (não versionados)
data/opportunities.json  tabela estruturada de cursos/prazos
run_ingest.py      CLI: percorre os editais e indexa
ask.py             CLI: pergunta -> resposta ancorada citando o edital
run_bot.py         sobe o bot do Telegram
docker-compose.yml weaviate + rabbitmq + redis
```

## Como rodar

```bash
cp .env.example .env      # preencha VOYAGE_API_KEY e GROQ_API_KEY
docker compose up -d weaviate rabbitmq redis
uv sync                   # ou pip install -e .
# coloque os PDFs em data/editais/
uv run python run_ingest.py
uv run python ask.py "como faço a inscrição no Sisu 2026?"
```

Rodando fora do docker (do host), aponte para as portas mapeadas no compose:

```bash
# no .env:
WEAVIATE_HTTP_URL=http://localhost:8081
WEAVIATE_GRPC_PORT=50052
```

PDFs são extraídos via LibreOffice quando disponível; sem ele, cai no fallback pypdf automaticamente.

### Bot do Telegram

```bash
# no .env, além de VOYAGE_API_KEY e GROQ_API_KEY:
TELEGRAM_BOT_TOKEN=<token do @BotFather>

docker compose up -d redis   # sessão de conversa fica no Redis
uv run python run_bot.py
```

`channels/engine.py` é o adaptador que liga o canal ao RAG de verdade
(`retrieval/generate.py`); `channels/fake_engine.py` continua no repo só
para testar o canal sem depender do RAG.

## Testes

```bash
uv run pytest tests/ -q   # puros: não tocam Weaviate, Voyage nem Groq
```

CI (`.github/workflows/ci.yml`) roda essa mesma suíte a cada push/PR em `main` — não precisa de nenhuma chave de API configurada como secret.

## Reaproveitamento

O core de RAG vem de uma plataforma de produção. Os módulos de extração, limpeza, chunking, embedding e o store idempotente do Weaviate são reaproveitados; a fonte de documentos foi trocada de storage remoto para leitura de arquivo local, e a camada de canal e recomendação é nova.

## Próximos passos (fases da PoC)

- [x] Fase 0: chunk + embed no `pipeline.py` e retrieval validado de ponta a ponta nos primeiros editais
- [x] Fase 1: tabela estruturada de oportunidades (`data/opportunities.json`) + filtro geo e temporal (`recommend/`)
- [ ] Fase 2: diálogo de perfil (structured output) + prompt de redação acessível
- [x] Fase 3: canal do Telegram + sessão no Redis + ligação ao RAG real (`channels/engine.py`)
- [ ] Fase 4: orquestração dos dois motores + testes de recusa
