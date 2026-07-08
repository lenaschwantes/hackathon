from __future__ import annotations

import logging
import os
import subprocess

logger = logging.getLogger(__name__)


def convert_to_docx(input_path: str, output_dir: str) -> str:
    """Convert a document (PDF, DOC, PPTX, etc.) to DOCX via LibreOffice."""
    result = subprocess.run(
        [
            "libreoffice",
            "--headless",
            "--convert-to",
            "docx",
            input_path,
            "--outdir",
            output_dir,
        ],
        check=False,
        capture_output=True,
        text=True,
    )
    if result.returncode != 0:
        stderr = (result.stderr or "").strip()
        stdout = (result.stdout or "").strip()
        detail = stderr or stdout or f"exit code {result.returncode}"
        raise RuntimeError(f"LibreOffice DOCX conversion failed: {detail}")

    base = os.path.splitext(os.path.basename(input_path))[0]
    docx_path = os.path.join(output_dir, f"{base}.docx")
    if not os.path.isfile(docx_path):
        raise FileNotFoundError(
            f"Expected DOCX output at {docx_path} after converting {input_path}"
        )
    return docx_path
