"""Testes dos módulos puros da ingestão.

Rodam sem infra (sem Weaviate, sem Voyage, sem docker). São o teste que
tu roda a cada commit para garantir que a base não quebrou.

    python -m pytest tests/test_pure.py -v
"""

import pytest

from ingestion.clean import clean_text
from utils.hashing import sha256_bytes
from utils.validation import assert_allowed_filename, assert_has_extractable_text


class TestCleanText:
    def test_colapsa_espacos_e_apara(self):
        assert clean_text("  olá    mundo  ") == "olá mundo"

    def test_normaliza_quebras_de_linha(self):
        assert clean_text("a\r\nb\rc") == "a\nb\nc"

    def test_colapsa_multiplas_linhas_em_branco(self):
        assert clean_text("a\n\n\n\n\nb") == "a\n\nb"

    def test_remove_caracteres_de_controle(self):
        assert clean_text("edital\x00 IFSC\x07") == "edital IFSC"

    def test_string_vazia_ou_none(self):
        assert clean_text("") == ""
        assert clean_text(None) == ""  # type: ignore[arg-type]


class TestHashing:
    def test_valor_conhecido(self):
        # sha256 de b"" é constante e documentado
        assert sha256_bytes(b"") == (
            "e3b0c44298fc1c149afbf4c8996fb924"
            "27ae41e4649b934ca495991b7852b855"
        )

    def test_idempotente(self):
        conteudo = b"edital IFSC 2026"
        assert sha256_bytes(conteudo) == sha256_bytes(conteudo)

    def test_conteudos_diferentes_geram_hash_diferente(self):
        assert sha256_bytes(b"edital A") != sha256_bytes(b"edital B")


class TestValidation:
    def test_aceita_extensoes_permitidas(self):
        assert assert_allowed_filename("edital.pdf") == ".pdf"
        assert assert_allowed_filename("EDITAL.DOCX") == ".docx"

    def test_rejeita_extensao_nao_suportada(self):
        with pytest.raises(ValueError):
            assert_allowed_filename("edital.txt")

    def test_texto_suficiente_passa(self):
        assert assert_has_extractable_text("x" * 150, min_chars=100) == 150

    def test_texto_curto_levanta_erro(self):
        with pytest.raises(ValueError):
            assert_has_extractable_text("curto", min_chars=100)
