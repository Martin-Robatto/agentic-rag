from typing import Any, Dict
from graph.state import GraphState
from ingestion import get_retriever


def retrieve(state: GraphState) -> Dict[str, Any]:
    print(f"Retrieving documents ...")

    question = state["question"]
    documents = get_retriever().invoke(question)

    return {
        "documents": documents,
        "question": question,
    }
