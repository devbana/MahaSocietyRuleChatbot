from __future__ import annotations

from pathlib import Path

from .config import DATA_DIR, UPLOAD_DIR


def ensure_directories() -> None:
    """Create the data directories needed by uploads and indexing."""
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    UPLOAD_DIR.mkdir(parents=True, exist_ok=True)


def save_uploaded_files(uploaded_files: list[object]) -> list[Path]:
    """Persist uploaded PDF files into the application upload directory."""
    saved_paths: list[Path] = []
    for uploaded_file in uploaded_files:
        if uploaded_file is None:
            continue
        temp_path = Path(getattr(uploaded_file, "name", str(uploaded_file)))
        file_path = UPLOAD_DIR / temp_path.name
        file_path.write_bytes(temp_path.read_bytes())
        saved_paths.append(file_path)
    return saved_paths


def list_available_pdf_paths() -> list[Path]:
    """Return every PDF path available for indexing from bundled and uploaded data."""
    pdf_paths = sorted(DATA_DIR.glob("*.pdf")) + sorted(UPLOAD_DIR.glob("*.pdf"))
    return pdf_paths


def list_available_pdfs() -> list[str]:
    """Return PDF filenames for display in the Gradio sidebar."""
    return [path.name for path in list_available_pdf_paths()]
