from langchain_core.prompts import PromptTemplate


def get_prompts():
    summarizer_prompt = PromptTemplate(
        template="""
You are an expert video summarizer.

Generate a clear and structured summary.

Rules:
- Only use the transcript
- 5–6 bullet points
- No assumptions

Transcript:
{context}
""",
        input_variables=["context"]
    )

    qna_prompt = PromptTemplate(
        template="""
Answer the question:

Context:
{context}

Question:
{question}

Rules:
- Be precise

Answer:
""",
        input_variables=["context", "question"]
    )

    return summarizer_prompt, qna_prompt
