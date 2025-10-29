from typing import Any, Dict
from graph.chains.generation import generation_chain
from graph.state import GraphState

def generate(state: GraphState) -> Dict[str, Any]:
    """
    Generates a response to the question.
    """
    print("Generating response ...")
    question = state["question"]
    documents = state["documents"]
    result = generation_chain.invoke({"question": question, "context": documents})
    return {
        "response": result,
        "question": question,
        "generation": result,
    }