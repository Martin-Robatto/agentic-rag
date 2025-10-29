from dotenv import load_dotenv

load_dotenv()
from typing import Any, Dict
import sys
import os

sys.path.append(
    os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
)
from graph.state import GraphState
from langchain.schema import Document
from langchain_tavily import TavilySearch

web_search_tool = TavilySearch(max_results=3)


def web_search(state: GraphState) -> Dict[str, Any]:
    """
    Performs a web search for the question.

    Args:
    state (dict): The current graph state.

    Returns:
    state (dict): The current graph state with the web search results.
    """

    print("Let me search the web for you ...")

    question = state["question"]
    if "documents" in state:
        documents = state["documents"]
    else:
        documents = None
    results = web_search_tool.invoke({"query": question})
    result_doc = "\n".join(result["content"] for result in results["results"])
    document = Document(page_content=result_doc)
    if documents:
        documents.append(document)
    else:
        documents = [document]
    return {
        "documents": documents,
        "question": question,
    }


if __name__ == "__main__":
    web_search({"question": "What is DRS?", "documents": None})
