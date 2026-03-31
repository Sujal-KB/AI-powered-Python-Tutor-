import streamlit as st
import warnings
warnings.filterwarnings('ignore')

from langchain_core.runnables import RunnableLambda, RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser

from src.ingestion import get_transcript, split_text
from src.chains import get_prompts
from src.llm import get_llm
from src.vector_database import create_vectordb
from src.router import get_llm_router
from src.py_compiler import get_py_output
from src.translator import translate_chunks



st.set_page_config(page_title="Youtube Chatbot", layout='wide')
st.title("AI Powered Youtube Based Python Learning Application.")

if "video_link" not in st.session_state:
    st.session_state.video_link = None

if "docs" not in st.session_state:
    st.session_state.docs = None

if "retriever" not in st.session_state:
    st.session_state.retriever = None


def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)


@st.cache_resource
def load_llm():
    llm = get_llm()
    llm.invoke("hello")
    return llm


@st.cache_data(show_spinner=False)
def process_video(video_link):
    text = get_transcript(video_link)

    if not text:
        return None

    text = text[:15000]

    chunks = split_text(text)

    with st.spinner("Translating video content..."):
        translated_chunks = translate_chunks(chunks)

    return translated_chunks


video_link = st.sidebar.text_input("Enter Youtube Video Link here:")
upload = st.sidebar.button("Upload", type='primary')

if upload and video_link:

    if st.session_state.video_link != video_link:

        with st.spinner("Processing video..."):

            docs = process_video(video_link)

            if docs:
                # create_vectordb not decorated with @st.cache_resource
                # because docs (list of Documents) is unhashable — guarded via session_state
                retriever = create_vectordb(docs)

                st.session_state.docs = docs
                st.session_state.retriever = retriever
                st.session_state.video_link = video_link

                st.sidebar.success("Video processed!")
            else:
                st.sidebar.error("Could not fetch transcript for this video.")

st.divider()

col1, col2 = st.columns(2)

with col1:
    st.write("#### Uploaded Video")
    if st.session_state.video_link:
        st.video(st.session_state.video_link)

with col2:

    tab1, tab2, tab3, tab4, tab5 = st.tabs(
        ['Key Notes', 'Python Compiler', 'Summarizer', 'MCQ Generator', 'QnA']
    )

    if st.session_state.docs:

        docs = st.session_state.docs
        retriever = st.session_state.retriever
        llm = load_llm()

        summarizer_prompt, qna_prompt = get_prompts()
        router_chain = get_llm_router(llm)

        summarizer_chain = (
            RunnableLambda(lambda _: docs)
            | RunnableLambda(format_docs)
            | summarizer_prompt
            | llm
            | StrOutputParser()
        )

        qna_chain = (
            {
                'context': retriever | RunnableLambda(format_docs),
                'question': RunnablePassthrough()
            }
            | qna_prompt
            | llm
            | StrOutputParser()
        )

    with tab1:
        key_points = st.text_area("Notebook for Key points", height=400)
        st.download_button("Download", key_points, "key_points.txt")

    with tab2:
        c1, c2 = st.columns(2)
        with c1:
            st.write("Python Compiler.")
        with c2:
            compile = st.button("Compile", type='primary')

        code = st.text_area("Enter Your Python Code.", height=350)

        if compile:
            with st.spinner("Compiling..."):
                output = get_py_output(code)
                st.write("Code Output:")
                st.success(output)

    with tab3:
        if st.button("Summarize Video"):

            if not st.session_state.docs:
                st.warning("⚠️ Upload video first")
            else:
                with st.spinner("Generating summary..."):
                    response = summarizer_chain.invoke("summarize")
                    st.success(response)

    with tab4:
        if st.button("Generate MCQs"):

            if not st.session_state.docs:
                st.warning("⚠️ Upload video first")
            else:
                with st.spinner("Generating MCQs..."):

                    mcq_prompt = f"""
Generate 5 multiple choice questions from the following content:

{format_docs(docs)}

Format:
Question:

a)
b)
c)
d)

After all the questions provide the answers:
Answers:

1.
2.
3.
4.
5.
"""
                    response = llm.invoke(mcq_prompt)
                    st.info(response.content)

    with tab5:
        user_query = st.text_input("Enter your query")

        if st.button("Get Response"):

            if not st.session_state.docs:
                st.warning("⚠️ Upload video first")
            else:
                with st.spinner("Generating answer..."):

                    intent = router_chain.invoke(user_query)

                    if intent == "summary":
                        response = summarizer_chain.invoke("summarize")
                        st.success(response)
                    else:  # "qa" or any unexpected output → default to RAG
                        response = qna_chain.invoke(user_query)
                        st.write(response)
