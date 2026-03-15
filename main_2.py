import os
from pathlib import Path
import sys
from dotenv import load_dotenv


from langchain_astradb import AstraDBVectorStore
from langchain.chat_models import init_chat_model
# from langchain_community.document_loaders import PyPDFLoader
# from langchain_core.documents import Document
from langchain_huggingface import ChatHuggingFace, HuggingFaceEmbeddings, HuggingFaceEndpoint
# from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_classic.chains import create_retrieval_chain
from langchain_classic.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate, PromptTemplate
from langchain_core.runnables import (RunnablePassthrough,)
from typing import List
from langchain_core.output_parsers import StrOutputParser
from langchain_community.document_loaders import DirectoryLoader, PyMuPDFLoader
from langchain_core.messages import HumanMessage, AIMessage
from langchain_core.documents import Document
from langchain_text_splitters import CharacterTextSplitter
import src.housing_society_law_assistant.config as config


def format_docs(docs: List[Document]):
    """Format documents for insertion into prompt"""
    formatted = []
    for i, doc in enumerate(docs):
        source = doc.metadata.get('source', 'Unknown')
        formatted.append(f"Document {i+1} (Source: {source}):\n{doc.page_content}")
    return "\n\n".join(formatted)



class MahaSocietyLawsLoader:
    def __init__(self, pdf_loader, directory, chunk_size, chunk_overlap, embedding_model):
        self.pdf_loader = pdf_loader
        self.directory = directory
        self.chunk_size = int(chunk_size)
        self.chunk_overlap = int(chunk_overlap)
        self.embedding_model = embedding_model
        self.embedding_status = False
        self.llm = init_chat_model(f"google_genai:{os.getenv('GEMINI_MODEL')}", api_key=os.getenv('GEMINI_KEY'))
        self.retriever = self.vector_store(chunk_=None).as_retriever(
            search_type="similarity",
            search_kwargs={"k": config.TOP_K}
        )


    def maha_doc_load(self):
        dir_loader = DirectoryLoader(self.directory,
                                     glob='*.pdf',
                                     loader_cls=self.pdf_loader,
                                     show_progress=True)
        dir_documents = dir_loader.load()
        return dir_documents

    def doc_chunker(self, docs):

        char_splitter = CharacterTextSplitter(separator='\n',
                                              chunk_size=self.chunk_size,
                                              chunk_overlap=self.chunk_overlap,
                                              length_function=len)
        char_chunks = char_splitter.split_documents(docs)
        return char_chunks

    def vector_store(self, chunk_):

        vector_store = AstraDBVectorStore(
            embedding=self.embedding_model,
            api_endpoint=os.getenv('ASTRA_DB_API_ENDPOINT'),
            collection_name=config.COLLECTION_NAME,
            token=os.getenv('ASTRA_DB_APPLICATION_TOKEN'),
            namespace=os.getenv('ASTRA_DB_NAMESPACE'),

        )

        if self.embedding_status:
            vector_store.add_documents(documents=chunk_)

        return vector_store


    def create_simple_rag(self):
        simple_prompt = PromptTemplate.from_template("""Answer the question based only on the following context:
                Context: {context}

                Question: {input}

                Answer:""")

        # LCEL
        # simple_rag_chain = (
        #         {"context": retriever | format_docs, "question": RunnablePassthrough()}
        #         | simple_prompt
        #         | llm
        #         | StrOutputParser()
        #
        # )

        ### Create stuff Docuemnt Chain
        document_chain = create_stuff_documents_chain(llm=self.llm, prompt=simple_prompt)

        ## create Full rAg chain
        rag_chain = create_retrieval_chain(retriever=self.retriever, combine_docs_chain=document_chain)
        return rag_chain

    def conversational_chat(self):

        # chat_prompt_ = ChatPromptTemplate.from_messages([
        #     ("system", "You are an expert on Maharashtra housing society laws."),
        #     ("placeholder", "{chat_history}"),
        #     ("human", "{input}")
        # ])

        chat_prompt_ = ChatPromptTemplate.from_messages([
            ("system", "You are an expert on Maharashtra housing society laws."),
            ("placeholder", "{chat_history}"),
            ("human", "Context: {context}\n\nQuestion: {input}"),
        ])
        document_chain = create_stuff_documents_chain(llm=self.llm, prompt=chat_prompt_)

        ## create Full rAg chain
        rag_chain = create_retrieval_chain(retriever=self.retriever, combine_docs_chain=document_chain)
        return rag_chain



if __name__ == '__main__':
    BASE_DIR = Path(__file__).resolve().parent

    load_dotenv(os.path.join(BASE_DIR,'.env'))

    print('Starting RAG System')
    embeddings = HuggingFaceEmbeddings(model=config.EMBEDDING_MODEL)

    obj_ = MahaSocietyLawsLoader(pdf_loader=PyMuPDFLoader,directory=config.DATA_DIR,  chunk_size=os.getenv('CHUNK_SIZE'), chunk_overlap=os.getenv('CHUNK_OVERLAP'),
                                 embedding_model=embeddings)

    if obj_.embedding_status:
        print('Reading PDFS')
        docs = obj_.maha_doc_load()
        print(f'Loaded {len(docs)} documents')
        # Chunker
        print('Converting Documents to chunks')
        chunks_ = obj_.doc_chunker(docs)
        print(f'Loaded {len(chunks_)} chunks')
        # for i in range(3):
        #     print(f'Chunk {i} : {chunks_[i]}')
        #     i += 1
        # Embedding and Vector Store
        print('Storing chunks into vector store')
        vector_store_ = obj_.vector_store(chunks_)
        print('Cheecking Similarity search')
        results = vector_store_.similarity_search(
            "Rules for Tenant for Parking",
                k=int(os.getenv('TOP_K'))
        )
        for res in results:
            print(f'* "{res.page_content}", metadata={res.metadata}')
    else:
        # vector_store_ = obj_.vector_store(None)
        # query_ = 'Rules for Tenant Parking in society'
        # Simple RAG
        # simple_rag = obj_.create_simple_rag()
        # response = simple_rag.invoke({'input': query_})
        #
        # for i, doc in enumerate(response["context"]):
        #     print(f"\nDoc {i + 1}: {doc.page_content}")

        chat_history = []

        chat_rag = obj_.conversational_chat()

        while True:
            query = input("\nYou: ")

            if query.lower() in ["exit", "quit"]:
                break

            response = chat_rag.invoke({
                "input": query,
                "chat_history": chat_history
            })

            answer = response["answer"]

            chat_history.append(HumanMessage(content=query))
            chat_history.append(AIMessage(content=answer))

            print("AI:", answer)
        print('All Good')

