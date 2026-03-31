def summarize_chunks(docs, llm, prompt):
    summaries = []

    for doc in docs:
        res = llm.invoke(
            prompt.format(context=doc.page_content)
        )
        summaries.append(res.content)

    final_summary = llm.invoke(
        "Combine and summarize the following:\n\n" +
        "\n\n".join(summaries)
    )

    return final_summary.content


def generate_mcq_chunks(docs, llm):
    mcqs = []

    for doc in docs:
        res = llm.invoke(
            f"""
Generate 2 multiple choice questions from the content below.

Content:
{doc.page_content}

Format:
Q:
a)
b)
c)
d)

Answer:
"""
        )
        mcqs.append(res.content)

    return "\n\n".join(mcqs)


def ask_question_rag(query, retriever, llm, prompt, format_docs):
    docs = retriever.invoke(query)
    context = format_docs(docs)

    response = llm.invoke(
        prompt.format(context=context, question=query)
    )

    return response.content