from graph.chains.retrieval_grader import retrieval_grader
from graph.chains.generation import generation_chain
from ingestion import get_retriever


def test_retrieval_grader_answer_true() -> None:
    question = "What is DRS?"
    retriever = get_retriever()
    docs = retriever.invoke(question)
    result = retrieval_grader.invoke({"question": question, "document": docs[0].page_content})
    assert result.isRelevant == True

def test_retrieval_grader_answer_false() -> None:
    question = "What is the capital of France?"
    retriever = get_retriever()
    docs = retriever.invoke(question)
    result = retrieval_grader.invoke({"question": question, "document": docs[0].page_content})
    assert result.isRelevant == False

def test_generation_chain() -> None:
    question = "What is DRS?"
    documents = get_retriever().invoke(question)
    result = generation_chain.invoke({"question": question, "context": documents[0].page_content})
    assert result != None