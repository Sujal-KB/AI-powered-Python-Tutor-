from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
import streamlit as st


@st.cache_resource
def load_embedding_model():
    return HuggingFaceEmbeddings(
        model_name="all-MiniLM-L6-v2"
    )


@st.cache_resource
def create_vectordb(chunks):
    embedding_model = load_embedding_model()

    vector_store = FAISS.from_documents(
        embedding=embedding_model,
        documents=chunks
    )

    retriever = vector_store.as_retriever(
        search_type="mmr",
        search_kwargs={"k": 8}
    )

    return retriever