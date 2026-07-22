"""Configuração central do Decifra.

Lê variáveis de ambiente via pydantic-settings. Nenhum segredo tem valor
padrão: chaves de API ficam vazias e devem ser preenchidas no ``.env``.
"""

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    # Weaviate
    weaviate_http_url: str = "http://weaviate:8080"
    weaviate_grpc_port: int = 50051
    raw_collection_suffix: str = "_raw"
    default_collection: str = "Editais"

    # Chunking
    chunk_size: int = 400
    chunk_overlap: int = 60
    data_dir: str = "data/editais"
    min_extracted_chars: int = 100

    # Embeddings (Voyage)
    voyage_model: str = "voyage-3"
    voyage_api_key: str = ""

    # LLM (Anthropic) -- a chave (ANTHROPIC_API_KEY) é lida direto do
    # ambiente pelo SDK oficial, não precisa de campo aqui. Dois modelos:
    # geração (respostas pro cidadão) usa mais capacidade; extração
    # (structured output mecânico) usa um modelo mais leve.
    anthropic_model_geracao: str = "claude-sonnet-5"
    anthropic_model_extracao: str = "claude-haiku-4-5"

    # Broker (opcional, para ingestão assíncrona via Celery)
    rabbitmq_url: str = "amqp://guest:guest@rabbitmq:5672//"
    celery_queue: str = "data_ingestion"

    # Ingestão automática (crawler de editais do IFSC)
    auto_ingest_ciclo_segundos: int = 60 * 60 * 24  # diário

    # Rate limiting (proteção de flood na entrada do canal)
    rate_limit_mensagens: int = 5
    rate_limit_janela_segundos: int = 10
    rate_limit_dedup_segundos: int = 5

    # Retrieval híbrido
    search_k: int = 12
    search_alpha: float = 0.6

    # Cache semântico do RAG (evita busca + geração pra perguntas
    # repetidas ou parafraseadas -- ver infra/semantic_cache.py)
    rag_cache_habilitado: bool = True
    rag_cache_limiar_similaridade: float = 0.90
    rag_cache_ttl_segundos: int = 60 * 60 * 3  # 3h
    # Recusa ("não encontrei essa informação") tem TTL bem mais curto --
    # o crawler roda periodicamente, então uma recusa de ontem pode não
    # valer mais hoje se um edital novo entrou na base.
    rag_cache_ttl_recusa_segundos: int = 60 * 30  # 30min
    rag_cache_max_itens: int = 80

    model_config = SettingsConfigDict(
        env_file=".env", env_file_encoding="utf-8", extra="ignore"
    )


settings = Settings()  # type: ignore
