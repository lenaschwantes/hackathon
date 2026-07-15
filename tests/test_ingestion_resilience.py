from __future__ import annotations

from ingestion.embeddings import VoyageEmbedding
from ingestion.pipeline import ingest_document
from ingestion.runner import executar_ciclo


class _StubEmbeddingFunction:
    def __init__(self):
        self.calls = 0

    def embed_documents(self, chunks):
        self.calls += 1
        if self.calls < 3:
            raise RuntimeError("rate limit reached")
        return [[0.1, 0.2] for _ in chunks]


class _DummyStore:
    collection_name = "TestCollection"

    def find_by_file_hash(self, file_hash):
        return None

    def upsert_document(self, **kwargs):
        return {"status": "upserted"}

    def insert_chunks(self, **kwargs):
        return None


def test_vectorize_documents_retries_transient_errors(monkeypatch):
    embedder = VoyageEmbedding.__new__(VoyageEmbedding)
    embedder.embedding_function = _StubEmbeddingFunction()
    embedder.embedding_model = "voyage-test"
    embedder.max_retries = 3
    embedder.retry_base_delay_seconds = 0
    embedder.max_delay_seconds = 0
    monkeypatch.setattr("ingestion.embeddings.time.sleep", lambda *_args, **_kwargs: None)

    vectors = embedder.Vectorize_documents(["chunk-a", "chunk-b"], batch_size=2)

    assert vectors == [[0.1, 0.2], [0.1, 0.2]]
    assert embedder.embedding_function.calls == 3


def test_ingest_document_returns_failed_on_extract_error(monkeypatch):
    monkeypatch.setattr("ingestion.pipeline.extract_text", lambda *args, **kwargs: (_ for _ in ()).throw(RuntimeError("PDF broken")))
    monkeypatch.setattr("ingestion.pipeline.clean_text", lambda text: text)
    monkeypatch.setattr("ingestion.pipeline.chunk_text", lambda text: [])

    resultado = ingest_document("broken.pdf", b"abc", _DummyStore())

    assert resultado["status"] == "failed"
    assert resultado["phase"] == "extract"


def test_executar_ciclo_returns_per_file_summary(monkeypatch):
    class _FakeStore:
        def find_by_storage_path(self, _path):
            return None

    monkeypatch.setattr("ingestion.runner.descobrir_novos", lambda *args, **kwargs: [object()])
    monkeypatch.setattr("ingestion.runner.ingerir_edital", lambda ref, store: {"status": "ok", "file_name": "ok.pdf"})
    monkeypatch.setattr("ingestion.runner.time.sleep", lambda *_args, **_kwargs: None)

    resumo = executar_ciclo(_FakeStore(), _FakeStore())

    assert resumo["sucesso"] == 1
    assert resumo["falha"] == 0
    assert resumo["arquivos"][0]["status"] == "ok"
    assert resumo["arquivos"][0]["file_name"] == "ok.pdf"
