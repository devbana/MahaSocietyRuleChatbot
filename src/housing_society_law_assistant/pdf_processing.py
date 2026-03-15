from __future__ import annotations

from langchain_community.document_loaders import PyMuPDFLoader
from langchain_core.documents import Document
from langchain_text_splitters import CharacterTextSplitter

from .config import CHUNK_OVERLAP, CHUNK_SIZE
from .document_store import list_available_pdf_paths


def load_pdf_documents() -> list[Document]:
    """Load all available PDFs and attach source metadata to each page document."""
    documents: list[Document] = []
    for pdf_path in list_available_pdf_paths():
        loader = PyMuPDFLoader(str(pdf_path))
        loaded = loader.load()
        for document in loaded:
            document.metadata["source"] = pdf_path.name
        documents.extend(loaded)
    return documents


def chunk_documents(documents: list[Document]) -> list[Document]:
    """Split PDF page documents into retrieval-sized chunks."""
    splitter = CharacterTextSplitter(
        separator="\n",
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        length_function=len,
    )
    return splitter.split_documents(documents)
