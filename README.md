# Housing Society Law Assistant

This project is a small RAG application for Maharashtra housing society documents. It lets you upload PDF rule books, index them, and ask plain-English questions through a Gradio chat interface.

The goal is simple: instead of manually scanning long society law PDFs, you can ask a question like "Can a tenant use the society parking space?" and get an answer grounded in the indexed documents, along with citations and source files.

## What It Does

- Uploads housing society law PDFs through the UI
- Extracts and chunks PDF text for retrieval
- Stores embeddings in either local Chroma or Astra DB
- Answers questions with Gemini using retrieved document context
- Shows cited sections and source documents beside the chat

## How The App Works

1. PDF files are loaded from `data/` and `data/uploads/`.
2. The text is split into chunks.
3. Those chunks are embedded with `sentence-transformers/all-MiniLM-L6-v2`.
4. Embeddings are stored in Chroma or Astra DB, depending on config.
5. When a user asks a question, the app retrieves the most relevant chunks.
6. Gemini generates the answer using only that retrieved context.

## Tech Stack

- UI: Gradio
- LLM: Google Gemini
- Embeddings: Hugging Face sentence transformers
- Vector store: Chroma or Astra DB
- PDF parsing: PyMuPDF / PyPDF via LangChain loaders

## Project Layout

- `app.py` - Gradio app entrypoint
- `main.py` - CLI/debug workflow for indexing and querying
- `data/` - bundled PDF documents
- `data/uploads/` - PDFs uploaded from the UI
- `vector_db/` - local Chroma persistence
- `src/housing_society_law_assistant/config.py` - environment-driven settings
- `src/housing_society_law_assistant/pdf_processing.py` - PDF loading and chunking
- `src/housing_society_law_assistant/vector_store.py` - Chroma/Astra integration
- `src/housing_society_law_assistant/qa_service.py` - retrieval and answer generation
- `src/housing_society_law_assistant/document_store.py` - upload and directory helpers

## Setup

### 1. Create a virtual environment

```bash
python -m venv .venv
source .venv/bin/activate
```

### 2. Install dependencies

```bash
pip install -r requirements.txt
```

### 3. Create your environment file

```bash
cp .env.example .env
```

Then fill in the values in `.env`.

Minimum config for local Chroma:

```env
GEMINI_KEY=your_gemini_api_key
GEMINI_MODEL=gemini-2.0-flash
VECTOR_STORE_BACKEND=chroma
EMBEDDING_MODEL=sentence-transformers/all-MiniLM-L6-v2
CHUNK_SIZE=1200
CHUNK_OVERLAP=200
TOP_K=5
```

If you want Astra DB instead of local storage:

```env
VECTOR_STORE_BACKEND=astra
ASTRA_DB_API_ENDPOINT=https://<your-db-endpoint>
ASTRA_DB_APPLICATION_TOKEN=AstraCS:...
ASTRA_DB_NAMESPACE=
ASTRA_DB_ENVIRONMENT=
```

## Run The App

```bash
python app.py
```

This starts the Gradio interface locally. From there you can:

- upload PDFs
- save them into `data/uploads/`
- index all available documents
- ask questions in chat

## CLI Mode

There is also a simple CLI flow in `main.py`.

List available PDFs:

```bash
python main.py list
```

Index documents:

```bash
python main.py index
```

Ask a question:

```bash
python main.py ask --query "What are the rules for parking allocation?"
```

Force a full Astra rebuild:

```bash
python main.py index --reindex
```

## Notes

- `VECTOR_STORE_BACKEND=astra` is the default in the current config.
- If you prefer fully local development, switch it to `chroma`.
- Re-indexing rebuilds the vector store from all PDFs currently available.
- Uploaded files are not enough by themselves; you still need to click `Index Documents`.
- Answers are only as good as the indexed PDF content. If the source material is incomplete, the answer will be incomplete too.

## Security

- Keep `.env` out of version control.
- Rotate API keys immediately if they were ever committed.
- Use `.env.example` only for placeholders, never real secrets.

## Future Improvements

- Better citation formatting for long answers
- Source-aware answer confidence
- OCR support for scanned PDFs
- Conversation memory with stricter legal grounding
- Deployment-ready config for hosted usage
