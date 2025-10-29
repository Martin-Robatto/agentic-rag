from langchain_core.prompts import ChatPromptTemplate
from pydantic import BaseModel, Field
from langchain_openai import ChatOpenAI

llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)


class GradeDocuments(BaseModel):
    """
    Binary score for relevance check on retrieved documents.
    """

    isRelevant: bool = Field(
        description="Whether the documents are relevant to the question"
    )


structured_llm = llm.with_structured_output(GradeDocuments)
system_prompt = """
You are a grader assessing relevance of a retrieved document to a user question. \n 
    If the document contains keyword(s) or semantic meaning related to the question, grade it as relevant. \n
    Give a binary score true or false to indicate whether the document is relevant to the question.
"""
user_prompt = """
Question: {question}
Documents: {documents}
"""

grade_prompt = ChatPromptTemplate.from_messages(
    [
        ("system", system_prompt),
        ("human", "Retrieved document: \n\n {document} \n\n User question: {question}"),
    ]
)

retrieval_grader = grade_prompt | structured_llm
