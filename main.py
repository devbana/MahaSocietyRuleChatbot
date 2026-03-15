from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

from dotenv import load_dotenv
from langchain_astradb import AstraDBVectorStore
from langchain_community.document_loaders import PyPDFLoader
from langchain_core.documents import Document
from langchain_huggingface import ChatHuggingFace, HuggingFaceEmbeddings, HuggingFaceEndpoint
from langchain_text_splitters import RecursiveCharacterTextSplitter


BASE_DIR = Path(__file__).resolve().parent
load_dotenv(BASE_DIR / ".env")

DATA_DIR = BASE_DIR / "data"
UPLOAD_DIR = DATA_DIR / "uploads"
COLLECTION_NAME = os.getenv("ASTRA_COLLECTION_NAME", "housing_society_laws")
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "sentence-transformers/all-MiniLM-L6-v2")
LLM_MODEL = os.getenv("LLM_MODEL", "HuggingFaceH4/zephyr-7b-beta")
CHUNK_SIZE = int(os.getenv("CHUNK_SIZE", "1200"))
CHUNK_OVERLAP = int(os.getenv("CHUNK_OVERLAP", "200"))
TOP_K = int(os.getenv("TOP_K", "5"))
ASTRA_DB_API_ENDPOINT = os.getenv("ASTRA_DB_API_ENDPOINT", "").strip()
ASTRA_DB_APPLICATION_TOKEN = os.getenv("ASTRA_DB_APPLICATION_TOKEN", "").strip()
ASTRA_DB_NAMESPACE = os.getenv("ASTRA_DB_NAMESPACE", "").strip() or None
ASTRA_DB_ENVIRONMENT = os.getenv("ASTRA_DB_ENVIRONMENT", "").strip() or None
HUGGINGFACEHUB_API_TOKEN = os.getenv("HUGGINGFACEHUB_API_TOKEN", "").strip()

PROMPT_TEMPLATE = """You are a legal assistant for Maharashtra housing society rules.
Answer the user using only the retrieved context.
If the answer is not fully supported by the context, say so clearly.

User question:
{question}

Retrieved context:
{context}

Return a concise answer followed by a short uncertainty note if needed.
"""


def fail(message: str) -> None:
    print(f"[error] {message}", file=sys.stderr)
    raise SystemExit(1)


def validate_env() -> None:
    missing = []
    if not ASTRA_DB_API_ENDPOINT:
        missing.append("ASTRA_DB_API_ENDPOINT")
    if not ASTRA_DB_APPLICATION_TOKEN:
        missing.append("ASTRA_DB_APPLICATION_TOKEN")
    if not HUGGINGFACEHUB_API_TOKEN:
        missing.append("HUGGINGFACEHUB_API_TOKEN")
    if missing:
        fail(f"Missing required environment variables: {', '.join(missing)}")


def list_pdf_paths() -> list[Path]:
    data_pdfs = sorted(DATA_DIR.glob("*.pdf"))
    upload_pdfs = sorted(UPLOAD_DIR.glob("*.pdf"))
    pdf_paths = data_pdfs + upload_pdfs
    print(f"[debug] Found {len(pdf_paths)} PDF(s)")
    for path in pdf_paths:
        print(f"[debug] PDF: {path}")
    return pdf_paths


def load_documents(pdf_paths: list[Path]) -> list[Document]:
    documents: list[Document] = []
    for pdf_path in pdf_paths:
        print(f"[debug] Loading PDF: {pdf_path.name}")
        loaded_docs = PyPDFLoader(str(pdf_path)).load()
        for doc in loaded_docs:
            doc.metadata["source"] = pdf_path.name
        documents.extend(loaded_docs)
    print(f"[debug] Loaded {len(documents)} page document(s)")
    return documents


def chunk_documents(documents: list[Document]) -> list[Document]:
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        separators=["\n\n", "\n", ". ", " "],
    )
    chunks = splitter.split_documents(documents)
    print(f"[debug] Split into {len(chunks)} chunk(s)")
    return chunks


def get_embeddings() -> HuggingFaceEmbeddings:
    print(f"[debug] Initializing embedding model: {EMBEDDING_MODEL}")
    return HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL)


def get_vector_store(embeddings: HuggingFaceEmbeddings, pre_delete_collection: bool = False) -> AstraDBVectorStore:
    print("[debug] Connecting to Astra DB")
    print(f"[debug] collection={COLLECTION_NAME}")
    print(f"[debug] namespace={ASTRA_DB_NAMESPACE!r}")
    print(f"[debug] environment={ASTRA_DB_ENVIRONMENT!r}")
    return AstraDBVectorStore(
        collection_name=COLLECTION_NAME,
        embedding=embeddings,
        token=ASTRA_DB_APPLICATION_TOKEN,
        api_endpoint=ASTRA_DB_API_ENDPOINT,
        namespace=ASTRA_DB_NAMESPACE,
        environment=ASTRA_DB_ENVIRONMENT,
        pre_delete_collection=pre_delete_collection,
    )


def preview_query_embedding(embeddings: HuggingFaceEmbeddings, query: str) -> None:
    vector = embeddings.embed_query(query)
    print(f"[debug] Query embedding length: {len(vector)}")
    print(f"[debug] Query embedding preview: {vector[:8]}")


def build_index(force_reindex: bool) -> None:
    pdf_paths = list_pdf_paths()
    if not pdf_paths:
        fail("No PDFs found under ./data or ./data/uploads")

    documents = load_documents(pdf_paths)
    if not documents:
        fail("PDFs were found but no text could be extracted")

    chunks = chunk_documents(documents)
    embeddings = get_embeddings()
    store = get_vector_store(embeddings, pre_delete_collection=force_reindex)

    print("[debug] Writing chunks to Astra DB")
    store.add_documents(chunks)
    print("[debug] Index build complete")


def retrieve(query: str) -> list[Document]:
    embeddings = get_embeddings()
    preview_query_embedding(embeddings, query)
    store = get_vector_store(embeddings, pre_delete_collection=False)
    print(f"[debug] Running similarity search with top_k={TOP_K}")
    documents = store.similarity_search(query, k=TOP_K)
    print(f"[debug] Retrieved {len(documents)} document(s)")
    for index, doc in enumerate(documents, start=1):
        source = doc.metadata.get("source", "unknown")
        page = doc.metadata.get("page", "n/a")
        excerpt = " ".join(doc.page_content.split())[:200]
        print(f"[debug] Match {index}: source={source} page={page} excerpt={excerpt}")
    return documents


def build_context(documents: list[Document]) -> str:
    sections = []
    for index, doc in enumerate(documents, start=1):
        source = doc.metadata.get("source", "unknown")
        page = doc.metadata.get("page", "n/a")
        sections.append(f"[Section {index}] Source: {source}, page {page}\n{doc.page_content}")
    return "\n\n".join(sections)


def get_llm() -> ChatHuggingFace:
    print(f"[debug] Initializing LLM endpoint: {LLM_MODEL}")
    endpoint = HuggingFaceEndpoint(
        repo_id=LLM_MODEL,
        task="text-generation",
        max_new_tokens=512,
        temperature=0.1,
    )
    return ChatHuggingFace(llm=endpoint)


def ask(query: str) -> None:
    documents = retrieve(query)
    if not documents:
        fail("No documents retrieved for the query")

    context = build_context(documents)
    prompt = PROMPT_TEMPLATE.format(question=query, context=context)

    print("[debug] Sending prompt to LLM")
    llm = get_llm()
    response = llm.invoke(prompt)

    print("\n=== FINAL ANSWER ===")
    print(response.content)
    print("\n=== SOURCES ===")
    for doc in documents:
        print(f"- {doc.metadata.get('source', 'unknown')} | page {doc.metadata.get('page', 'n/a')}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Astra DB RAG debug runner")
    parser.add_argument(
        "command",
        choices=["list", "index", "ask"],
        help="list PDFs, index PDFs into Astra, or ask a question",
    )
    parser.add_argument(
        "--query",
        help="Question to ask when using the 'ask' command",
    )
    parser.add_argument(
        "--reindex",
        action="store_true",
        help="Delete and rebuild the Astra collection before indexing",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    validate_env()

    if args.command == "list":
        list_pdf_paths()
        return

    if args.command == "index":
        build_index(force_reindex=args.reindex)
        return

    if args.command == "ask":
        if not args.query:
            fail("--query is required for the 'ask' command")
        ask(args.query)
        return

    fail(f"Unsupported command: {args.command}")


if __name__ == "__main__":
    main()
