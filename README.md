# Decifra

Assistente conversacional que traduz editais da educação pública para linguagem simples, ancorado em RAG. Entregue via Telegram, orienta o cidadão sobre cursos e formas de ingresso no IFSC.

Projeto da 1ª Jornada Incubintech, Inovação Aberta, Desafio 12 (LabCiDig).

## Arquitetura

O perfil do cidadão alimenta dois motores separados, e eles não se misturam:

- **Recomendação (estruturado):** qual curso, câmpus, inscrição aberta ou não. Sai de uma consulta com filtro por localização, nível de curso e modalidade, cortada pelo calendário. O LLM não decide data, recebe o resultado pronto.
- **Tradução (RAG):** o que significa a cota, requisitos, forma de ingresso. Sai da recuperação híbrida ancorada no edital, sempre citando a fonte.

Essa separação é o que garante fidelidade: o assistente nunca inventa prazo, e quando não há base no acervo, reconhece que não sabe em vez de chutar.

`channels/engine.py` orquestra os dois: assim que o perfil fica completo (cidade, escolaridade, interesse e nível de curso desejado), chama a Recomendação (`dialogue/recommendation.py` -> `recommend/opportunities.py`); mensagens seguintes, com o perfil já completo, seguem para a Tradução (RAG) — a menos que a pessoa peça outra opção explicitamente ("mostra outra opção"), caso em que um classificador leve via LLM (`quer_nova_recomendacao`) detecta o pedido e gera nova recomendação em vez de cair no RAG. Um motor nunca redige com dado do outro. A extração de perfil também recebe o histórico recente da conversa como contexto, pra resolver respostas curtas que só fazem sentido junto da pergunta anterior (ex: "advogado" respondendo "qual seu interesse?").

Stack: embeddings com Voyage (`voyage-3`), geração com Anthropic (Claude Sonnet 5 na redação pro cidadão, Claude Haiku 4.5 na extração/classificação estruturada), store vetorial no Weaviate com busca híbrida (vetor + BM25).

## Estrutura

```
config/            settings central (Weaviate, Voyage, Anthropic, chunking, retrieval)
ingestion/         extração, limpeza, chunking, embedding e store no Weaviate
ingestion/sources/ fontes de editais (crawler do site do IFSC, pasta local, fallback)
retrieval/         busca híbrida (vetor + BM25) + geração ancorada via Anthropic
recommend/         motor estruturado: filtro de oportunidades por perfil
channels/          adaptadores de canal (Telegram) + sessão no Redis + engine.py (liga o canal ao RAG)
utils/             hashing e validação
tests/             testes puros (rodam sem infra e sem chaves de API)
data/editais/      PDFs dos editais do IFSC (não versionados)
data/opportunities.json  tabela estruturada de cursos/prazos
run_ingest.py      CLI: percorre a pasta local e indexa (ingestão manual)
run_auto_ingest.py CLI: descobre editais no site do IFSC e indexa (ingestão automática)
ask.py             CLI: pergunta -> resposta ancorada citando o edital
run_bot.py         sobe o bot do Telegram
docker-compose.yml weaviate + rabbitmq + redis
```

## Como rodar

```bash
cp .env.example .env      # preencha VOYAGE_API_KEY e ANTHROPIC_API_KEY
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

### Ingestão automática (crawler do IFSC)

```bash
uv run python run_auto_ingest.py            # roda um ciclo e termina
uv run python run_auto_ingest.py --loop     # roda continuamente (intervalo em
                                             # settings.auto_ingest_ciclo_segundos,
                                             # default diário)
```

Descobre editais nas páginas públicas do IFSC (`editais-com-inscricoes-abertas`
e `-encerradas`), captura o status de onde cada um foi listado (nunca infere
por data) e ingere só os que ainda não foram processados (detecção
incremental por `storage_path`). A fonte HTML fica isolada atrás da
interface `EditalSource` (`ingestion/sources/base.py`) — se o site mudar de
estrutura ou cair, `FallbackEditalSource` cai automaticamente pra
`LocalFolderSource` (a mesma pasta local de sempre), sem tocar no pipeline.
Cada edital é processado isoladamente: falha em um (download quebrado, PDF
corrompido) tem retry com backoff exponencial (até 3x) e não derruba os
outros — na última falha vira um "dead-letter" no log.

### Bot do Telegram

```bash
# no .env, além de VOYAGE_API_KEY e ANTHROPIC_API_KEY:
TELEGRAM_BOT_TOKEN=<token do @BotFather>

docker compose up -d redis   # sessão de conversa fica no Redis
uv run python run_bot.py
```

`channels/engine.py` é o adaptador que liga o canal ao RAG de verdade
(`retrieval/generate.py`); `channels/fake_engine.py` continua no repo só
para testar o canal sem depender do RAG. `channels/rate_limit.py` limita
10 mensagens por usuário a cada 60s (contador no Redis) — protege as
chamadas pagas à Anthropic/Voyage de flood.

## Testes

```bash
uv run pytest tests/ -q   # puros: não tocam Weaviate, Voyage nem Anthropic
```

CI (`.github/workflows/ci.yml`) roda essa mesma suíte a cada push/PR em `main` — não precisa de nenhuma chave de API configurada como secret.

## Reaproveitamento

O core de RAG vem de uma plataforma de produção. Os módulos de extração, limpeza, chunking, embedding e o store idempotente do Weaviate são reaproveitados; a fonte de documentos foi trocada de storage remoto para leitura de arquivo local, e a camada de canal e recomendação é nova.

## Próximos passos (fases da PoC)

- [x] Fase 0: chunk + embed no `pipeline.py` e retrieval validado de ponta a ponta nos primeiros editais
- [x] Fase 1: tabela estruturada de oportunidades (`data/opportunities.json`) + filtro geo e temporal (`recommend/`)
- [x] Fase 2: diálogo de perfil (structured output) + prompt de redação acessível
- [x] Fase 3: canal do Telegram + sessão no Redis + ligação ao RAG real (`channels/engine.py`)
- [x] Fase 4: orquestração dos dois motores + testes de recusa

## Diferenciais (além das fases obrigatórias)

- **CI** (`.github/workflows/ci.yml`): roda a suíte de testes puros a cada push/PR.
- **Rate limiting** (`channels/rate_limit.py`): protege as chamadas pagas à Anthropic/Voyage de flood por usuário.
- **Ingestão automática** (`ingestion/sources/`, `run_auto_ingest.py`): acaba com a inserção manual de editais, com fonte HTML isolada do pipeline e fallback pra pasta local.
