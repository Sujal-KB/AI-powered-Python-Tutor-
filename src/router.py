from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnablePassthrough, RunnableLambda
from langchain_core.output_parsers import StrOutputParser


def get_router_prompt():
    return PromptTemplate(
        template="""
You are an intent classifier.

Classify the user query into:
- "summary" → summary, key points, overview
- "qa" → specific question

Return ONLY one word.

Query:
{query}
""",
        input_variables=["query"]
    )


def get_llm_router(llm):
    router_prompt = get_router_prompt()

    router_chain = (
        {"query": RunnablePassthrough()}
        | router_prompt
        | llm
        | StrOutputParser()
        | RunnableLambda(lambda x: x.strip().lower())
    )

    return router_chain
