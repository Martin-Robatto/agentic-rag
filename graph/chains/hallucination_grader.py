from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnableSequence
from langchain_openai import ChatOpenAI
from pydantic import BaseModel, Field

llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)


class GradeHallucination(BaseModel):
    """
    Binary score for hallucination check on generated responses.
    """

    is_hallucinated: bool = Field(
        description="Whether the response is hallucinated grounded in the retrieved documents"
    )


structured_llm = llm.with_structured_output(GradeHallucination)
system_prompt = """
You are a grader assessing hallucination of a generated response based on the retrieved documents. \n 
    If the response is not grounded in the retrieved documents, grade it as hallucinated. \n
    Give a binary score true or false to indicate whether the response is hallucinated.
"""

prompt = ChatPromptTemplate.from_messages(
    [
        ("system", system_prompt),
        (
            "human",
            "Retrieved documents: \n\n {documents} \n\n Generated response: {response}",
        ),
    ]
)

hallucination_grader = prompt | structured_llm