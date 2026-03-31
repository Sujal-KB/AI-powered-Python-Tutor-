from langchain_openai import ChatOpenAI
from dotenv import load_dotenv
import os
import streamlit as st

load_dotenv()

git_token = os.getenv("GITHUB_TOKEN")


@st.cache_resource
def get_llm():
    llm = ChatOpenAI(
        api_key=git_token,
        base_url="https://models.github.ai/inference",
        model='openai/gpt-4o-mini',
        temperature=1.5
    )

    return llm
