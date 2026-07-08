"""Configuração central do Decifra.

Lê variáveis de ambiente via pydantic-settings. Nenhum segredo tem valor
padrão: chaves de API ficam vazias e devem ser preenchidas no ``.env``.
"""

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    # Weaviate
    weaviate_http_url: str = "http://weaviate:8080"
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

    # LLM (Anthropic)
    anthropic_model: str = "claude-haiku-4-5-20251001"
    anthropic_api_key: str = ""

    # Broker (opcional, para ingestão assíncrona via Celery)
    rabbitmq_url: str = "amqp://guest:guest@rabbitmq:5672//"
    celery_queue: str = "data_ingestion"

    # Retrieval híbrido
    search_k: int = 12
    search_alpha: float = 0.6

    model_config = SettingsConfigDict(
        env_file=".env", env_file_encoding="utf-8", extra="ignore"
    )


settings = Settings()  # type: ignore
