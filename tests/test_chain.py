from graph.chains.hallucination_grader import hallucination_grader
from graph.chains.retrieval_grader import retrieval_grader
from graph.chains.generation import generation_chain
from graph.chains.router import question_router
from ingestion import get_retriever


def test_retrieval_grader_answer_true() -> None:
    question = "What is DRS?"
    retriever = get_retriever()
    docs = retriever.invoke(question)
    result = retrieval_grader.invoke(
        {"question": question, "document": docs[0].page_content}
    )
    assert result.isRelevant == True


def test_retrieval_grader_answer_false() -> None:
    question = "What is the capital of France?"
    retriever = get_retriever()
    docs = retriever.invoke(question)
    result = retrieval_grader.invoke(
        {"question": question, "document": docs[0].page_content}
    )
    assert result.isRelevant == False


def test_generation_chain() -> None:
    question = "What is DRS?"
    documents = get_retriever().invoke(question)
    result = generation_chain.invoke(
        {"question": question, "context": documents[0].page_content}
    )
    assert result != None

def test_hallucination_grader_false() -> None:
    question = "What is DRS?"
    documents = get_retriever().invoke(question)
    generation = generation_chain.invoke(
        {"question": question, "context": documents}
    )
    result = hallucination_grader.invoke(
        {"documents": documents[0].page_content, "response": generation}
    )
    assert result.is_hallucinated == False

def test_hallucination_grader_true() -> None:
    question = "What is DRS?"
    documents = get_retriever().invoke(question)
    generation = "DRS is a platform for data science and machine learning."
    result = hallucination_grader.invoke(
        {"question": question, "documents": documents, "response": generation}
    )
    assert result.is_hallucinated == True

def test_question_router_vectorstore() -> None:
        question = "What is DRS?"
        result = question_router.invoke({"question": question})
        assert result.datasource == "vectorstore"

def test_question_router_websearch() -> None:
    question = "What is the capital of France?"
    result = question_router.invoke({"question": question})
    assert result.datasource == "websearch"