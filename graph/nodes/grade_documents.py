from typing import Any, Dict
from graph.chains.retrieval_grader import retrieval_grader
from graph.state import GraphState

def grade_documents(state: GraphState) -> Dict[str, Any]:
    """
    Determines whether the documents are relevant to the question.
    If any document is not relevant, we will set the web_search flag to True.
    
    Args:
    state (dict): The current graph state.

    Returns:
    state (dict): Filter out irrelevant documents and set the web_search flag to True if any document is not relevant.
    """
    print(f"Grading documents ...")
    question = state["question"]
    documents = state["documents"]
    relevant_documents = []
    web_search = False
    for document in documents:
        result = retrieval_grader.invoke({"question": question, "document": document.page_content})
        if result.isRelevant:
            relevant_documents.append(document)
        else:
            web_search = True
    return {
        "documents": relevant_documents,
        "question": question,
        "web_search": web_search,
    }