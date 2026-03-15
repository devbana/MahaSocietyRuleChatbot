# Housing Society Law Assistant

Gradio application for querying uploaded housing society law PDFs using retrieval-augmented generation with Hugging Face models for both embeddings and answer generation.

## Features

- Upload PDF documents from the Gradio sidebar
- Index documents into a local Chroma vector database or Astra DB
- Ask legal questions in a chat interface
- Retrieve the top 5 relevant chunks for each query
- Use Hugging Face embeddings for semantic search
- Use a Hugging Face LLM for final answer generation
- Display the generated answer, cited legal sections, and source documents

## Setup

1. Create and activate a virtual environment.
2. Install dependencies:

```bash
pip install -r requirements.txt
```

3. Create a `.env` file from the example and set your Hugging Face token:

```bash
cp .env.example .env
```

Then edit `.env` and set:

```bash
HUGGINGFACEHUB_API_TOKEN=your_token_here
```

For local development, the default vector backend is Chroma:

```bash
VECTOR_STORE_BACKEND=chroma
```

To use Astra DB instead, set:

```bash
VECTOR_STORE_BACKEND=astra
ASTRA_DB_API_ENDPOINT=https://<db-id>-<region>.apps.astra.datastax.com
ASTRA_DB_APPLICATION_TOKEN=AstraCS:...
ASTRA_DB_NAMESPACE=default_keyspace
ASTRA_DB_ENVIRONMENT=prod
```

4. Run the app:

```bash
python app.py
```

## Project Structure

- `app.py`: Gradio entrypoint
- `src/housing_society_law_assistant/config.py`: application settings
- `src/housing_society_law_assistant/document_store.py`: upload persistence
- `src/housing_society_law_assistant/pdf_processing.py`: PDF loading and chunking
- `src/housing_society_law_assistant/vector_store.py`: Chroma/Astra indexing and retrieval
- `src/housing_society_law_assistant/qa_service.py`: answer generation and citation shaping

## Function Summary

- `ensure_directories()`: creates the data and upload directories used by the app.
- `save_uploaded_files()`: copies uploaded PDFs into the local upload folder.
- `list_available_pdf_paths()` / `list_available_pdfs()`: discover bundled and uploaded PDFs for indexing and display.
- `load_pdf_documents()`: reads each PDF and converts pages into LangChain `Document` objects.
- `chunk_documents()`: splits loaded pages into smaller chunks for retrieval.
- `get_embeddings()`: creates the Hugging Face embedding model client.
- `get_vector_store()`: opens the configured Chroma or Astra vector collection.
- `vector_store_has_data()`: checks whether indexing has already been done.
- `build_vector_store()`: rebuilds the full Chroma index from all available PDFs.
- `retrieve_sections()`: runs semantic similarity search for the user query.
- `format_context()`: converts retrieved chunks into prompt context.
- `build_citations()`: creates citation metadata for the UI.
- `unique_sources()`: extracts distinct source document names.
- `get_llm()`: creates the Hugging Face LLM client.
- `answer_question()`: performs retrieval-augmented generation and returns answer, citations, and sources.
- `upload_documents()`: Gradio handler for saving files from the sidebar.
- `index_documents()`: Gradio handler for building the vector database.
- `ask_question()`: Gradio chat handler for answering user queries.
- `create_app()`: builds the Gradio interface and wires button actions.

## Execution Hierarchy

1. `create_app()` initializes the UI and ensures the data directories exist.
2. `upload_documents()` stores new PDFs in `data/uploads`.
3. `index_documents()` triggers `build_vector_store()`.
4. `build_vector_store()` calls `load_pdf_documents()` and then `chunk_documents()`.
5. `get_embeddings()` and `get_vector_store()` prepare Chroma with Hugging Face embeddings.
6. Chunks are stored either in the local Chroma database under `data/vector_db` or in the configured Astra DB collection.
7. `ask_question()` validates the request and calls `answer_question()`.
8. `answer_question()` calls `retrieve_sections()` to fetch the most relevant chunks.
9. `format_context()` prepares the retrieved chunks for the prompt.
10. `get_llm()` creates the Hugging Face LLM client and generates the final answer.
11. `build_citations()` and `unique_sources()` prepare supporting evidence for the UI.
12. Gradio displays the answer, cited sections, and source list back to the user.

## Model Configuration

- Embedding model default: `sentence-transformers/all-MiniLM-L6-v2`
- LLM default: `HuggingFaceH4/zephyr-7b-beta`

You can change both defaults in `src/housing_society_law_assistant/config.py`.

## Astra DB Notes

- This project uses LangChain's `AstraDBVectorStore` integration when `VECTOR_STORE_BACKEND=astra`.
- Re-indexing with Astra DB recreates the configured collection before loading fresh chunks.
- You still need the Hugging Face token for answer generation, even when Astra DB is used for vector storage.
