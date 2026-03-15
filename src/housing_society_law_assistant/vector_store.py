from __future__ import annotations

import shutil

from langchain_core.documents import Document
from langchain_huggingface import HuggingFaceEmbeddings

from .config import (
    ASTRA_DB_API_ENDPOINT,
    ASTRA_DB_APPLICATION_TOKEN,
    ASTRA_DB_ENVIRONMENT,
    ASTRA_DB_NAMESPACE,
    COLLECTION_NAME,
    EMBEDDING_MODEL,
    TOP_K,
    VECTOR_DB_DIR,
    VECTOR_STORE_BACKEND,
)
from .pdf_processing import chunk_documents, load_pdf_documents


def get_embeddings() -> HuggingFaceEmbeddings:
    """Create the embedding client used for Chroma indexing and retrieval."""
    return HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL)


def validate_vector_store_config() -> None:
    """Validate the configured vector backend and its required environment."""
    if VECTOR_STORE_BACKEND == "chroma":
        return

    if VECTOR_STORE_BACKEND != "astra":
        raise RuntimeError(
            f"Unsupported VECTOR_STORE_BACKEND '{VECTOR_STORE_BACKEND}'. Use 'chroma' or 'astra'."
        )

    if not ASTRA_DB_API_ENDPOINT or not ASTRA_DB_APPLICATION_TOKEN:
        raise RuntimeError(
            "Astra DB requires ASTRA_DB_API_ENDPOINT and ASTRA_DB_APPLICATION_TOKEN."
        )


def get_vector_store(*, pre_delete_collection: bool = False):
    """Create the configured vector store for legal document chunks."""
    validate_vector_store_config()

    if VECTOR_STORE_BACKEND == "astra":
        try:
            from langchain_astradb import AstraDBVectorStore
        except ImportError as exc:
            raise RuntimeError(
                "Astra DB support is not installed. Run `pip install -r requirements.txt`."
            ) from exc

        return AstraDBVectorStore(
            collection_name=COLLECTION_NAME,
            embedding=get_embeddings(),
            token=ASTRA_DB_APPLICATION_TOKEN,
            api_endpoint=ASTRA_DB_API_ENDPOINT,
            environment=ASTRA_DB_ENVIRONMENT,
            namespace=ASTRA_DB_NAMESPACE,
            pre_delete_collection=pre_delete_collection,
        )

    from langchain_chroma import Chroma

    return Chroma(
        collection_name=COLLECTION_NAME,
        embedding_function=get_embeddings(),
        persist_directory=str(VECTOR_DB_DIR),
    )


def vector_store_has_data() -> bool:
    """Check whether the persisted vector store already contains indexed chunks."""
    validate_vector_store_config()

    if VECTOR_STORE_BACKEND == "chroma":
        if not VECTOR_DB_DIR.exists():
            return False
        store = get_vector_store()
        return store._collection.count() > 0

    store = get_vector_store()
    return bool(store.similarity_search("housing society law", k=1))


def build_vector_store() -> int:
    """Load PDFs, split them into chunks, and rebuild the persisted Chroma index."""
    documents = load_pdf_documents()
    chunks = chunk_documents(documents)

    if VECTOR_STORE_BACKEND == "chroma" and VECTOR_DB_DIR.exists():
        shutil.rmtree(VECTOR_DB_DIR)

    store = get_vector_store(pre_delete_collection=VECTOR_STORE_BACKEND == "astra")
    if chunks:
        store.add_documents(chunks)
    return len(chunks)


def retrieve_sections(query: str) -> list[Document]:
    """Fetch the top matching legal chunks for a user question."""
    store = get_vector_store()
    return store.similarity_search(query, k=TOP_K)
