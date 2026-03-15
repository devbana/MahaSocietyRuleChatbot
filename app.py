from __future__ import annotations

import inspect

import gradio as gr

from src.housing_society_law_assistant.config import APP_TITLE, COLLECTION_NAME, VECTOR_STORE_BACKEND
from src.housing_society_law_assistant.document_store import (
    ensure_directories,
    list_available_pdfs,
    save_uploaded_files,
)
from src.housing_society_law_assistant.qa_service import answer_question
from src.housing_society_law_assistant.vector_store import build_vector_store


CHATBOT_SUPPORTS_TYPE = "type" in inspect.signature(gr.Chatbot.__init__).parameters
ChatMessage = dict[str, str]


def get_storage_summary() -> str:
    """Explain where embeddings are stored for the active vector backend."""
    if VECTOR_STORE_BACKEND == "astra":
        return f"Embeddings are stored in Astra DB collection `{COLLECTION_NAME}`."
    return "Embeddings are stored in the local Chroma vector database."


def format_available_documents() -> str:
    """Render the current PDF inventory for the sidebar."""
    available_pdfs = list_available_pdfs()
    if not available_pdfs:
        return "No PDF documents available yet."
    return "\n".join(f"- {pdf_name}" for pdf_name in available_pdfs)


def upload_documents(files: list[object] | None) -> tuple[str, str]:
    """Store uploaded PDFs and refresh the visible document list."""
    if not files:
        return "No files selected.", format_available_documents()

    saved_paths = save_uploaded_files(files)
    if not saved_paths:
        return "No PDF files were saved.", format_available_documents()

    return (
        f"Saved {len(saved_paths)} PDF file(s). Use 'Index Documents' only if you want to add these PDFs to Astra DB.",
        format_available_documents(),
    )


def index_documents() -> str:
    """Build or rebuild the vector store from all available PDFs."""
    try:
        count = build_vector_store()
    except Exception as exc:
        return f"Indexing failed: {exc}"

    if count == 0:
        return "No PDF content was found to index."
    return f"Indexed {count} document chunk(s). Their embeddings are now stored in Astra DB collection `{COLLECTION_NAME}`."


def format_citations(citations: list[dict[str, str]]) -> str:
    """Render cited sections into a readable multiline text block."""
    if not citations:
        return "No cited legal sections available."
    return "\n\n".join(
        f"{citation['title']} | {citation['source']} | page {citation['page']}\n{citation['excerpt']}"
        for citation in citations
    )


def format_sources(sources: list[str]) -> str:
    """Render unique source filenames for the sources panel."""
    if not sources:
        return "No document sources available."
    return "\n".join(f"- {source}" for source in sources)


def extract_message_text(content: object) -> str:
    """Flatten Gradio message content into plain text."""
    if content is None:
        return ""
    if isinstance(content, str):
        return content
    if isinstance(content, dict):
        text = content.get("text")
        return text if isinstance(text, str) else ""
    if isinstance(content, list):
        parts: list[str] = []
        for item in content:
            text = extract_message_text(item)
            if text:
                parts.append(text)
        return "\n".join(parts)
    return str(content)


def normalize_history(history: list[ChatMessage] | None) -> list[ChatMessage]:
    """Normalize chat history into Gradio's messages format."""
    if not history:
        return []
    normalized: list[ChatMessage] = []
    for message in history:
        if not isinstance(message, dict):
            continue
        role = str(message.get("role", "")).strip()
        content = extract_message_text(message.get("content"))
        if role and content:
            normalized.append({"role": role, "content": content})
    return normalized


def append_chat_turn(
    history: list[ChatMessage],
    user_message: str,
    assistant_message: str,
) -> list[ChatMessage]:
    """Append a user/assistant pair to the Gradio messages history."""
    history.append({"role": "user", "content": user_message})
    history.append({"role": "assistant", "content": assistant_message})
    return history


def ask_question(
    message: str,
    history: list[ChatMessage] | None,
) -> tuple[list[ChatMessage], str, str, str]:
    """Handle a chat turn, including validation, retrieval, and UI formatting."""
    chat_history = normalize_history(history)
    if not message.strip():
        return chat_history, "Enter a legal question to continue.", "No document sources available.", ""

    try:
        response = answer_question(message, chat_history)
    except Exception as exc:
        error_message = f"Unable to answer the question: {exc}"
        append_chat_turn(chat_history, message, error_message)
        return chat_history, error_message, "No document sources available.", ""

    append_chat_turn(chat_history, message, response["answer"])
    return (
        chat_history,
        format_citations(response["citations"]),
        format_sources(response["sources"]),
        "",
    )


def create_app() -> gr.Blocks:
    """Create and wire the Gradio interface for upload, indexing, and Q&A."""
    ensure_directories()
    chatbot_kwargs = {
        "label": "Housing Society Law Chat",
        "height": 420,
    }
    # Force messages format for current Gradio versions.
    if CHATBOT_SUPPORTS_TYPE:
        chatbot_kwargs["type"] = "messages"

    with gr.Blocks(title=APP_TITLE) as demo:
        gr.Markdown(f"# {APP_TITLE}")
        with gr.Row():
            with gr.Column(scale=1, min_width=280):
                gr.Markdown("## Upload PDFs")
                gr.Markdown("## Vector Storage")
                gr.Markdown(get_storage_summary())
                gr.Markdown("If your embeddings are already in Astra DB, you can ask questions directly without re-indexing here.")
                file_input = gr.File(
                    label="Upload housing law PDFs",
                    file_count="multiple",
                    file_types=[".pdf"],
                )
                upload_button = gr.Button("Save Uploads", variant="secondary")
                upload_status = gr.Textbox(label="Upload Status", interactive=False)
                document_list = gr.Markdown(value=format_available_documents())
                index_button = gr.Button("Index Documents", variant="primary")
                index_status = gr.Textbox(label="Index Status", interactive=False)

            with gr.Column(scale=2):
                gr.Markdown("## Chat Interface")
                gr.Markdown("Questions are answered directly from PDF embeddings already stored in Astra DB.")
                chatbot = gr.Chatbot(**chatbot_kwargs)
                question_box = gr.Textbox(
                    label="Ask a question",
                    placeholder="Ask a question about housing society laws",
                )
                ask_button = gr.Button("Ask", variant="primary")
                gr.Markdown("## Answer Section")
                citations_box = gr.Textbox(
                    label="Cited Legal Sections",
                    lines=12,
                    interactive=False,
                )
                gr.Markdown("## Sources Section")
                sources_box = gr.Textbox(
                    label="Document Sources",
                    lines=6,
                    interactive=False,
                )

        upload_button.click(
            upload_documents,
            inputs=[file_input],
            outputs=[upload_status, document_list],
        )
        index_button.click(index_documents, outputs=[index_status])
        ask_button.click(
            ask_question,
            inputs=[question_box, chatbot],
            outputs=[chatbot, citations_box, sources_box, question_box],
        )
        question_box.submit(
            ask_question,
            inputs=[question_box, chatbot],
            outputs=[chatbot, citations_box, sources_box, question_box],
        )

    return demo


if __name__ == "__main__":
    app = create_app()
    app.launch()
