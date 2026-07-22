# Decifra

Assistente conversacional que traduz editais da educação pública para linguagem simples, ancorado em RAG. Entregue via Telegram, orienta o cidadão sobre cursos e formas de ingresso no IFSC.

Projeto da 1ª Jornada Incubintech, Inovação Aberta, Desafio 12 (LabCiDig).

## Arquitetura

O perfil do cidadão alimenta dois motores separados, e eles não se misturam:

- **Recomendação (estruturado):** qual curso, câmpus, inscrição aberta ou não. Sai de uma consulta com filtro por localização, nível de curso e modalidade, cortada pelo calendário. O LLM não decide data, recebe o resultado pronto.

- **Tradução (RAG):** o que significa a cota, requisitos, forma de ingresso. Sai da recuperação híbrida ancorada no edital, sempre citando a fonte.

Essa separação é o que garante fidelidade: o assistente nunca inventa prazo, e quando não há base no acervo, reconhece que não sabe em vez de chutar.

`channels/engine.py` orquestra os dois: assim que o perfil fica completo (cidade, escolaridade, interesse e nível de curso desejado), chama a Recomendação (`dialogue/recommendation.py` -> `recommend/opportunities.py`); mensagens seguintes, com o perfil já completo, seguem para a Tradução (RAG) — a menos que a pessoa peça outra opção explicitamente ("mostra outra opção"), caso em que um classificador leve via LLM (`quer_nova_recomendacao`) detecta o pedido e gera nova recomendação em vez de cair no RAG. Um motor nunca redige com dado do outro. A extração de perfil também recebe o histórico recente da conversa como contexto, pra resolver respostas curtas que só fazem sentido junto da pergunta anterior (ex: "advogado" respondendo "qual seu interesse?").

Campos de conjunto fechado (escolaridade, nível de curso, alcance) são coletados por teclado de botão inline, nunca por menu numerado em texto — o valor do botão vai direto pro perfil, sem passar pelo extrator (mais barato e determinístico). Escolaridade e nível de curso são coerentes entre si: quem já concluiu uma faculdade, por exemplo, nunca vê "Técnico integrado" como opção, e se só houver um nível plausível pra escolaridade informada, a pergunta de nível é pulada e preenchida direto (`dialogue/profile.py::niveis_compativeis`/`aplicar_coerencia_nivel`). A qualquer momento da conversa — mesmo no meio da coleta ou de uma pergunta ao RAG — a pessoa pode dizer "recomeçar" (ou variações como "começar de novo", `/recomecar`) pra reiniciar do zero; o pedido tem prioridade sobre o roteamento normal e sempre passa por confirmação antes de apagar os dados.

Stack: embeddings com Voyage (`voyage-3`), geração com Anthropic (Claude Sonnet 5 na redação pro cidadão, Claude Haiku 4.5 na extração/classificação estruturada), store vetorial no Weaviate com busca híbrida (vetor + BM25).

### Cache semântico do RAG

Antes de rodar busca híbrida + geração, `retrieval/generate.py::answer` consulta `infra/semantic_cache.py`: embeda a pergunta (Voyage, mesmo embedder do retrieval) e compara por similaridade de cosseno contra as últimas perguntas já respondidas (scan linear no Redis, sem índice vetorial — proporcional ao volume de um hackathon). Acima do limiar configurável (`rag_cache_limiar_similaridade`, default 0.90), devolve a resposta cacheada direto, sem rodar busca nem geração de novo. Recusa ("não encontrei essa informação") tem TTL bem mais curto que resposta com base real, já que o crawler roda periodicamente e a base muda. Redis ou embedder indisponível: falha aberta, o cache é só pulado, nunca quebra a resposta.

## Estrutura

```
config/            settings central (Weaviate, Voyage, Anthropic, chunking, retrieval, cache) + prompts dos LLMs
infra/             rate limiting e dedup de mensagem, cache semântico do RAG -- infraestrutura transversal, sem lógica de domínio
ingestion/         extração+conversão, limpeza, chunking, embedding e store no Weaviate
ingestion/sources/ fontes de editais (crawler do site do IFSC, pasta local, fallback)
retrieval/         busca híbrida (vetor + BM25) + geração ancorada via Anthropic
recommend/         motor estruturado: filtro de oportunidades por perfil (calendário, geografia, catálogo)
dialogue/          lógica de conversação: extração de perfil, intents, onboarding e reinício
channels/          adaptadores de canal (Telegram) + sessão no Redis + engine.py (liga o canal aos dois motores)
utils/             hashing e validação
tests/unit/        testes puros (sem infra, sem chave de API) das funções isoladas
tests/integration/ testes de orquestração (engine, telegram, ingestão) -- a maioria também roda sem infra;
                    só os marcados `@pytest.mark.integration` exigem Weaviate/Voyage/Anthropic reais
tests/e2e/         golden dataset ponta a ponta (RAG + recomendação, com stubs de LLM/busca)
data/editais/      PDFs dos editais do IFSC (não versionados)
data/opportunities.json  tabela estruturada de cursos/prazos
data/calendario.json     calendário oficial de janelas de inscrição
deploy/            configurações de deploy (systemd service)
run_ingest.py      CLI: percorre a pasta local e indexa (ingestão manual)
run_auto_ingest.py CLI: descobre editais no site do IFSC e indexa (ingestão automática)
ask.py             CLI: pergunta -> resposta ancorada citando o edital
run_bot.py         sobe o bot do Telegram
docker-compose.yml infraestrutura (weaviate + rabbitmq + redis)
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
WEAVIATE_GRPC_PORT=50051
```

PDFs são extraídos via LibreOffice quando disponível; sem ele, cai no fallback pypdf automaticamente.

### Ingestão automática (crawler do IFSC)

```bash
uv run python run_auto_ingest.py            # roda um ciclo e termina
uv run python run_auto_ingest.py --loop     # roda continuamente (intervalo em
                                             # settings.auto_ingest_ciclo_segundos,
                                             # default diário)
```

Descobre editais nas páginas públicas do IFSC (`editais-com-inscricoes-abertas` e `-encerradas`), captura o status de onde cada um foi listado (nunca infere por data) e ingere só os que ainda não foram processados (detecção incremental por `storage_path`). A fonte HTML fica isolada atrás da interface `EditalSource` (`ingestion/sources/base.py`) — se o site mudar de estrutura ou cair, `FallbackEditalSource` cai automaticamente pra `LocalFolderSource` (a mesma pasta local de sempre), sem tocar no pipeline. Cada edital é processado isoladamente: falha em um (download quebrado, PDF corrompido) tem retry com backoff exponencial (até 3x) e não derruba os outros — na última falha vira um "dead-letter" no log.

### Bot do Telegram

```bash
# no .env, além de VOYAGE_API_KEY e ANTHROPIC_API_KEY:
TELEGRAM_BOT_TOKEN=<token do @BotFather>

docker compose up -d redis   # sessão de conversa fica no Redis
uv run python run_bot.py
```

`channels/engine.py` é o adaptador que liga o canal ao RAG de verdade (`retrieval/generate.py`); `channels/fake_engine.py` continua no repo só para testar o canal sem depender do RAG. `infra/rate_limit.py` limita 5 mensagens por usuário a cada 10s (contador no Redis) — protege as chamadas pagas à Anthropic/Voyage de flood, e também deduplica clique de botão/mensagem repetida na mesma janela curta. A qualquer momento, `/recomecar` (comando) ou "recomeçar" (texto livre) reinicia a conversa, com confirmação antes de apagar os dados.

## Testes

```bash
uv run pytest tests/ -q -m "not integration"   # não tocam Weaviate, Voyage nem Anthropic reais
uv run pytest tests/ -q                        # suíte completa, exige a infra real de pé (ver docker-compose.yml)
```

CI (`.github/workflows/ci.yml`) roda a primeira variante a cada push/PR em `main` — não precisa de nenhuma chave de API configurada como secret.

## Reaproveitamento

O core de RAG vem de uma plataforma de produção. Os módulos de extração, limpeza, chunking, embedding e o store idempotente do Weaviate são reaproveitados; a fonte de documentos foi trocada de storage remoto para leitura de arquivo local, e a camada de canal e recomendação é nova.

## Roadmap Futuro

- Teste de integração automatizado batendo no Weaviate real pra pipeline de ingestão completa (hoje as fontes, a descoberta incremental e o retry são cobertos isoladamente, sem infra real).

- Expansão do banco de dados `opportunities.json` com mais campi e modalidades.

- Adição de suporte a múltiplos idiomas no processamento de editais.

## Diferenciais (além das fases obrigatórias)

- **CI** (`.github/workflows/ci.yml`): roda a suíte de testes puros a cada push/PR.

- **Rate limiting e dedup** (`infra/rate_limit.py`): protege as chamadas pagas à Anthropic/Voyage de flood por usuário, em mensagem de texto e em clique de botão.

- **Cache semântico do RAG** (`infra/semantic_cache.py`): pergunta repetida ou parafraseada não roda busca nem geração de novo -- devolve a resposta já cacheada.

- **Ingestão automática** (`ingestion/sources/`, `run_auto_ingest.py`): acaba com a inserção manual de editais, com fonte HTML isolada do pipeline e fallback pra pasta local.

- **Coleta por botão com coerência** (`dialogue/profile.py`): campo de conjunto fechado nunca é menu numerado em texto: escolaridade e nível de curso ficam coerentes entre si (quem já tem faculdade não vê "Técnico integrado"), e reinício ("recomeçar") funciona em qualquer ponto da conversa, não só no menu pós-recomendação.

- **Dados Estruturados Ricos**: O projeto já conta com 18 oportunidades e 27 janelas de calendário cadastradas, cobrindo 16 campi e 4 níveis de ensino.
