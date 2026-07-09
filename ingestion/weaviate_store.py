"""Cliente Weaviate para a camada raw do pipeline de ingestão.

Suporta conexão local (SDK Python) e remota (REST/GraphQL). Oferece
deduplicação por ``file_hash`` e upsert de documentos com
``pipeline_status=raw_ready``.
"""

from __future__ import annotations

import json
import os
import uuid
from datetime import UTC, datetime
from urllib.parse import urlparse

import httpx
import weaviate
from weaviate.classes import init
from weaviate.classes.config import Configure, DataType, Property
from weaviate.classes.query import Filter

from config.settings import settings

PIPELINE_STATUS_RAW_READY = "raw_ready"
PIPELINE_STATUS_INDEXED = "indexed"
DEDUP_PIPELINE_STATUSES = frozenset({PIPELINE_STATUS_RAW_READY, PIPELINE_STATUS_INDEXED})

_LOCAL_WEAVIATE_HOSTS = frozenset({"weaviate", "localhost", "127.0.0.1"})


def raw_collection_name(logical_collection: str) -> str:
    """Deriva o nome da coleção raw a partir da coleção lógica.

    Parameters
    ----------
    logical_collection : str
        Nome lógico (ex.: ``Documentos``).

    Returns
    -------
    str
        Nome com sufixo ``_raw`` (ex.: ``Documentos_raw``), sem duplicar
        o sufixo se já presente.
    """
    suffix = settings.raw_collection_suffix
    if logical_collection.endswith(suffix):
        return logical_collection
    return f"{logical_collection}{suffix}"


class WeaviateStore:
    """Persistência e consulta de documentos na coleção raw do Weaviate.

    Parameters
    ----------
    collection_name : str
        Nome da classe/coleção Weaviate (ex.: ``Documentos_raw``).

    Attributes
    ----------
    collection_name : str
        Nome da coleção configurada.
    """

    def __init__(self, collection_name: str):
        self.collection_name = collection_name
        self._class_name = collection_name
        parsed = urlparse(settings.weaviate_http_url)
        self._host = parsed.hostname or "weaviate"
        self._base_url = settings.weaviate_http_url.rstrip("/")
        self._use_rest = self._host not in _LOCAL_WEAVIATE_HOSTS
        self.client = None if self._use_rest else self._connect()
        if self._use_rest:
            self._class_name = self._resolve_remote_class_name(collection_name)

    def close(self) -> None:
        if self.client:
            self.client.close()

    def _connect(self) -> weaviate.client.WeaviateClient:
        port = urlparse(settings.weaviate_http_url).port or 8080

        return weaviate.connect_to_local(
            host=self._host,
            port=port,
            additional_config=init.AdditionalConfig(
                timeout=init.Timeout(init=60),
            ),
            skip_init_checks=True,
        )

    def _rest_request(self, method: str, path: str, **kwargs) -> httpx.Response:
        with httpx.Client(timeout=60.0) as http:
            response = http.request(method, f"{self._base_url}{path}", **kwargs)
            response.raise_for_status()
            return response

    def _resolve_remote_class_name(self, collection_name: str) -> str:
        with httpx.Client(timeout=60.0) as http:
            response = http.get(f"{self._base_url}/v1/schema")
            response.raise_for_status()
            classes = response.json().get("classes", [])
            names = [item.get("class") for item in classes if item.get("class")]
            for name in names:
                if name.lower() == collection_name.lower():
                    return name
        return collection_name

    def _ensure_remote_collection(self) -> None:
        with httpx.Client(timeout=60.0) as http:
            response = http.get(f"{self._base_url}/v1/schema/{self._class_name}")
            if response.status_code == 404:
                raise RuntimeError(
                    f"Coleção '{self._class_name}' não existe no Weaviate remoto. "
                    "Crie-a na VM ou ajuste DEFAULT_COLLECTION/WEAVIATE_RAW_COLLECTION."
                )
            response.raise_for_status()

    def get_or_create_collection(self):
        if self._use_rest:
            self._ensure_remote_collection()
            return None

        if not self.client.is_connected():
            self.client.connect()

        if not self.client.collections.exists(self.collection_name):
            try:
                return self.client.collections.create(
                    self.collection_name,
                    properties=[
                        Property(name="content", data_type=DataType.TEXT),
                        Property(
                            name="file_name",
                            data_type=DataType.TEXT,
                            skip_vetorization=True,
                        ),
                        Property(
                            name="file_hash",
                            data_type=DataType.TEXT,
                            skip_vetorization=True,
                        ),
                        Property(
                            name="bucket",
                            data_type=DataType.TEXT,
                            skip_vetorization=True,
                        ),
                        Property(
                            name="storage_path",
                            data_type=DataType.TEXT,
                            skip_vetorization=True,
                        ),
                        Property(
                            name="content_type",
                            data_type=DataType.TEXT,
                            skip_vetorization=True,
                        ),
                        Property(
                            name="source_format",
                            data_type=DataType.TEXT,
                            skip_vetorization=True,
                        ),
                        Property(
                            name="converted_from",
                            data_type=DataType.TEXT,
                            skip_vetorization=True,
                        ),
                        Property(
                            name="extractor",
                            data_type=DataType.TEXT,
                            skip_vetorization=True,
                        ),
                        Property(
                            name="text_chars",
                            data_type=DataType.INT,
                            skip_vetorization=True,
                        ),
                        Property(
                            name="pipeline_status",
                            data_type=DataType.TEXT,
                            skip_vetorization=True,
                        ),
                        Property(
                            name="ingested_at",
                            data_type=DataType.TEXT,
                            skip_vetorization=True,
                        ),
                        Property(
                            name="chunk_index",
                            data_type=DataType.INT,
                            skip_vetorization=True,
                        ),
                    ],
                    vectorizer_config=[
                        Configure.NamedVectors.none(
                            name="content_vector",
                            vector_index_config=Configure.VectorIndex.hnsw(),
                        )
                    ],
                )
            except Exception as e:
                # condição de corrida: outro worker criou a coleção entre o exists() e o create()
                if "already exists" not in str(e).lower():
                    raise
        return self.client.collections.get(self.collection_name)

    def find_by_file_hash(self, file_hash: str) -> dict | None:
        """Busca documento existente por hash.

        Parameters
        ----------
        file_hash : str
            SHA-256 do conteúdo.

        Returns
        -------
        dict or None
            ``{'uuid': str, 'properties': dict}`` ou ``None`` se ausente.
        """
        if self._use_rest:
            return self._rest_find_by_file_hash(file_hash)

        collection = self.get_or_create_collection()
        result = collection.query.fetch_objects(
            filters=Filter.by_property("file_hash").equal(file_hash),
            limit=1,
        )
        if not result.objects:
            return None
        obj = result.objects[0]
        return {
            "uuid": str(obj.uuid),
            "properties": dict(obj.properties or {}),
        }

    def get_content_by_file_hash(self, file_hash: str) -> str | None:
        """Retorna o texto indexado para um hash.

        Parameters
        ----------
        file_hash : str
            SHA-256 do conteúdo.

        Returns
        -------
        str or None
            Campo ``content`` do objeto, ou ``None`` se não encontrado.
        """
        if self._use_rest:
            return self._rest_get_content_by_file_hash(file_hash)

        collection = self.get_or_create_collection()
        result = collection.query.fetch_objects(
            filters=Filter.by_property("file_hash").equal(file_hash),
            limit=1,
        )
        if not result.objects:
            return None
        props = dict(result.objects[0].properties or {})
        content = props.get("content")
        return content if isinstance(content, str) else None

    def _rest_get_content_by_file_hash(self, file_hash: str) -> str | None:
        self._ensure_remote_collection()
        safe_hash = json.dumps(file_hash)
        query = f"""
        {{
          Get {{
            {self._class_name}(
              where: {{ path: ["file_hash"], operator: Equal, valueText: {safe_hash} }}
              limit: 1
            ) {{
              content
            }}
          }}
        }}
        """
        payload = self._rest_request(
            "POST", "/v1/graphql", json={"query": query}
        ).json()
        items = (
            payload.get("data", {})
            .get("Get", {})
            .get(self._class_name, [])
            or []
        )
        if not items:
            return None
        content = items[0].get("content")
        return content if isinstance(content, str) else None

    def find_duplicate_by_file_hash(self, file_hash: str) -> dict | None:
        """Retorna documento se o hash já foi ingerido (raw_ready ou indexed).

        Usado pelo early dedup no pipeline — evita reprocessar conteúdo já
        presente no Weaviate.

        Parameters
        ----------
        file_hash : str
            SHA-256 do conteúdo.

        Returns
        -------
        dict or None
            Objeto existente com status deduplicável, ou ``None``.
        """
        existing = self.find_by_file_hash(file_hash)
        if not existing:
            return None
        status = (existing["properties"].get("pipeline_status") or "").lower()
        if status in DEDUP_PIPELINE_STATUSES:
            return existing
        return None

    def _rest_find_by_file_hash(self, file_hash: str) -> dict | None:
        self._ensure_remote_collection()
        safe_hash = json.dumps(file_hash)
        query = f"""
        {{
          Get {{
            {self._class_name}(
              where: {{ path: ["file_hash"], operator: Equal, valueText: {safe_hash} }}
              limit: 1
            ) {{
              file_hash
              pipeline_status
              _additional {{ id }}
            }}
          }}
        }}
        """
        payload = self._rest_request(
            "POST", "/v1/graphql", json={"query": query}
        ).json()
        items = (
            payload.get("data", {})
            .get("Get", {})
            .get(self._class_name, [])
            or []
        )
        if not items:
            return None
        item = items[0]
        obj_id = item.get("_additional", {}).get("id")
        return {
            "uuid": obj_id,
            "properties": {
                k: v for k, v in item.items() if k != "_additional"
            },
        }

    def upsert_document(
        self,
        *,
        content: str,
        file_name: str,
        file_hash: str,
        bucket: str,
        storage_path: str,
        content_type: str | None,
        text_chars: int,
        extractor: str,
        source_format: str | None,
        converted_from: str | None,
    ) -> dict:
        """Insere ou atualiza documento no Weaviate.

        Parameters
        ----------
        content : str
            Texto extraído e limpo.
        file_name : str
            Nome do arquivo.
        file_hash : str
            SHA-256 do conteúdo binário.
        bucket : str
            Origem do documento (ex: pasta local).
        storage_path : str
            Caminho no bucket.
        content_type : str or None
            MIME type do arquivo.
        text_chars : int
            Quantidade de caracteres extraídos.
        extractor : str
            Extrator utilizado (ex.: ``docx``, ``libreoffice``).
        source_format : str or None
            Formato de origem detectado.
        converted_from : str or None
            Formato convertido antes da extração.

        Returns
        -------
        dict
            ``uuid``, ``pipeline_status`` e ``skipped`` (``True`` se hash já
            existir com ``raw_ready`` ou ``indexed``).
        """
        existing = self.find_by_file_hash(file_hash)
        if existing:
            status = (existing["properties"].get("pipeline_status") or "").lower()
            if status in DEDUP_PIPELINE_STATUSES:
                return {
                    "uuid": existing["uuid"],
                    "pipeline_status": status,
                    "skipped": True,
                }

        ingested_at = datetime.now(UTC).isoformat()
        properties = {
            "content": content,
            "file_name": file_name,
            "file_hash": file_hash,
            "bucket": bucket,
            "storage_path": storage_path,
            "content_type": content_type or "",
            "source_format": source_format or "",
            "converted_from": converted_from or "",
            "extractor": extractor,
            "text_chars": text_chars,
            "pipeline_status": PIPELINE_STATUS_RAW_READY,
            "ingested_at": ingested_at,
        }

        if self._use_rest:
            return self._rest_upsert_document(existing, properties, file_hash)

        collection = self.get_or_create_collection()

        if existing:
            collection.data.update(uuid=existing["uuid"], properties=properties)
            return {
                "uuid": existing["uuid"],
                "pipeline_status": PIPELINE_STATUS_RAW_READY,
                "skipped": False,
            }

        doc_uuid = self._generate_chunk_uuid(file_hash, 0)
        collection.data.insert(properties=properties, uuid=doc_uuid)
        return {
            "uuid": doc_uuid,
            "pipeline_status": PIPELINE_STATUS_RAW_READY,
            "skipped": False,
        }

    def _rest_upsert_document(
        self,
        existing: dict | None,
        properties: dict,
        file_hash: str,
    ) -> dict:
        doc_uuid = existing["uuid"] if existing else self._generate_chunk_uuid(file_hash, 0)
        body = {
            "class": self._class_name,
            "id": doc_uuid,
            "properties": properties,
        }
        if existing:
            self._rest_request(
                "PUT",
                f"/v1/objects/{self._class_name}/{doc_uuid}",
                json=body,
            )
        else:
            self._rest_request("POST", "/v1/objects", json=body)

        return {
            "uuid": doc_uuid,
            "pipeline_status": PIPELINE_STATUS_RAW_READY,
            "skipped": False,
        }

    def insert_chunks(
        self,
        *,
        file_name: str,
        file_hash: str,
        chunks: list[str],
        vectors: list[list[float]],
    ) -> None:
        """
        Insere múltiplos chunks com batch insert e UUID determinístico.
        Usado quando o pipeline tiver chunking explícito implementado.
        """
        if len(chunks) != len(vectors):
            raise ValueError("chunks and vectors length mismatch")

        if self._use_rest:
            raise NotImplementedError("insert_chunks via REST ainda não implementado")

        collection = self.get_or_create_collection()

        with collection.batch.dynamic() as batch:
            for idx, (text, vec) in enumerate(zip(chunks, vectors, strict=True)):
                batch.add_object(
                    properties={
                        "content": text,
                        "file_name": file_name,
                        "file_hash": file_hash,
                        "chunk_index": idx,
                    },
                    vector={"content_vector": vec},
                    uuid=self._generate_chunk_uuid(file_hash, idx),
                )

    def _generate_chunk_uuid(self, file_hash: str, position: int) -> str:
        return str(uuid.uuid5(uuid.NAMESPACE_DNS, f"{file_hash}__{position}"))
