from __future__ import annotations

from collections import OrderedDict

from langchain.chat_models import init_chat_model
from langchain_classic.chains import create_retrieval_chain
from langchain_classic.chains.combine_documents import create_stuff_documents_chain
from langchain_core.documents import Document
from langchain_core.messages import AIMessage, HumanMessage
from langchain_core.prompts import ChatPromptTemplate

from .config import GEMINI_KEY, GEMINI_MODEL, TOP_K
from .vector_store import get_vector_store


def build_citations(documents: list[Document]) -> list[dict[str, str]]:
    """Shape retrieved chunks into a compact citation payload for the UI."""
    citations: list[dict[str, str]] = []
    for index, document in enumerate(documents, start=1):
        source = document.metadata.get("source", "Unknown source")
        page = str(document.metadata.get("page", "N/A"))
        excerpt = " ".join(document.page_content.strip().split())[:280]
        citations.append(
            {
                "title": f"Section {index}",
                "source": source,
                "page": page,
                "excerpt": excerpt + ("..." if len(excerpt) == 280 else ""),
            }
        )
    return citations


def unique_sources(documents: list[Document]) -> list[str]:
    """Return unique source filenames while preserving retrieval order."""
    return list(OrderedDict.fromkeys(document.metadata.get("source", "Unknown source") for document in documents))


def get_llm():
    """Create the Gemini chat client used for answer generation."""
    if not GEMINI_KEY:
        raise RuntimeError("GEMINI_KEY is not set.")

    return init_chat_model(f"google_genai:{GEMINI_MODEL}", api_key=GEMINI_KEY)


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


def build_chat_history(history: list[dict[str, str]] | None) -> list[HumanMessage | AIMessage]:
    """Convert Gradio message history into LangChain message objects."""
    if not history:
        return []

    messages: list[HumanMessage | AIMessage] = []
    for message in history:
        role = message.get("role")
        content = extract_message_text(message.get("content"))
        if not content:
            continue
        if role == "user":
            messages.append(HumanMessage(content=content))
        elif role == "assistant":
            messages.append(AIMessage(content=content))
    return messages


def get_chat_chain():
    """Create the conversational retrieval chain used by the Gradio app."""
    prompt = ChatPromptTemplate.from_messages(
        [
            ("system", "You are an expert on Maharashtra housing society laws."),
            ("placeholder", "{chat_history}"),
            ("human", "Context: {context}\n\nQuestion: {input}"),
        ]
    )
    document_chain = create_stuff_documents_chain(llm=get_llm(), prompt=prompt)
    retriever = get_vector_store().as_retriever(
        search_type="similarity",
        search_kwargs={"k": TOP_K},
    )
    return create_retrieval_chain(retriever=retriever, combine_docs_chain=document_chain)


def answer_question(question: str, history: list[dict[str, str]] | None = None) -> dict[str, object]:
    """Run conversational RAG and return answer metadata."""
    response = get_chat_chain().invoke(
        {
            "input": question,
            "chat_history": build_chat_history(history),
        }
    )
    retrieved_docs = response.get("context", [])
    answer = response.get("answer", "")

    return {
        "answer": answer,
        "citations": build_citations(retrieved_docs),
        "sources": unique_sources(retrieved_docs),
    }
