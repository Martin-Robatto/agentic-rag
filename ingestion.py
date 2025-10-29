import os
from dotenv import load_dotenv

load_dotenv()
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import WebBaseLoader
from langchain_openai import OpenAIEmbeddings
from langchain_pinecone import PineconeVectorStore


def ingest_documents():
    urls = [
        "https://rishivikram348.medium.com/formula-one-for-dummies-part-one-the-basics-of-the-sport-26de6eeeca38",
        "https://medium.com/@Formula.101/an-introduction-to-formula-1-teams-and-drivers-94de161ec82f",
        "https://en.wikipedia.org/wiki/Drag_reduction_system",
    ]

    print("Loading documents from URLs...")
    documents = [WebBaseLoader(url).load() for url in urls]
    documents_list = [item for sublist in documents for item in sublist]

    print("Splitting documents...")
    text_splitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(
        chunk_size=1000, chunk_overlap=200
    )
    doc_splits = text_splitter.split_documents(documents_list)

    print(f"Ingesting {len(doc_splits)} document chunks into Pinecone...")
    embeddings = OpenAIEmbeddings()
    index_name = os.getenv("PINECONE_INDEX_NAME")

    PineconeVectorStore.from_documents(
        documents=doc_splits, embedding=embeddings, index_name=index_name
    )

    print(f"✓ Successfully ingested {len(doc_splits)} documents into Pinecone")


def get_retriever():
    embeddings = OpenAIEmbeddings()
    index_name = os.getenv("PINECONE_INDEX_NAME")

    vector_store = PineconeVectorStore(embedding=embeddings, index_name=index_name)

    return vector_store.as_retriever(search_kwargs={"k": 5})
