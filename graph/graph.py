from dotenv import load_dotenv

from graph.chains.router import question_router

load_dotenv()
from langgraph.graph import StateGraph, START, END
from graph.consts import (
    RETRIEVE_NODE,
    WEB_SEARCH_NODE,
    GRADE_DOCUMENTS_NODE,
    GENERATE_NODE,
)
from graph.nodes import retrieve, web_search, grade_documents, generate
from graph.state import GraphState
from graph.chains.hallucination_grader import hallucination_grader
from graph.chains.answer_grader import answer_grader


def decide_to_generate(state: GraphState) -> StateGraph:
    if state["web_search"]:
        return WEB_SEARCH_NODE
    else:
        return GENERATE_NODE

def grade_generation_grounded_in_documents_and_question(state: GraphState) -> str:
    question = state["question"]
    documents = state["documents"]
    generation = state["generation"]
    result = hallucination_grader.invoke(
        {"documents": documents, "response": generation}
    )
    if result.is_hallucinated:
        return "Hallucinated"
    else:
        answer = answer_grader.invoke(
            {"question": question, "generation": generation}
        )
        if answer.is_correct:
            return "Correct"
        else:
            return "Incorrect"

def route_question(state: GraphState) -> str:
    question = state["question"]
    result = question_router.invoke({"question": question})
    print("Routing question to ", result.datasource)
    return result.datasource

workflow = StateGraph(GraphState)
workflow.add_node(RETRIEVE_NODE, retrieve)
workflow.add_node(WEB_SEARCH_NODE, web_search)
workflow.add_node(GRADE_DOCUMENTS_NODE, grade_documents)
workflow.add_node(GENERATE_NODE, generate)

workflow.add_conditional_edges(START, route_question, {"vectorstore": RETRIEVE_NODE, "websearch": WEB_SEARCH_NODE})
workflow.add_edge(RETRIEVE_NODE, GRADE_DOCUMENTS_NODE)
workflow.add_conditional_edges(
    GRADE_DOCUMENTS_NODE,
    decide_to_generate,
    {WEB_SEARCH_NODE: WEB_SEARCH_NODE, GENERATE_NODE: GENERATE_NODE},
)
workflow.add_conditional_edges(
    GENERATE_NODE,
    grade_generation_grounded_in_documents_and_question,
    {"Correct": END, "Incorrect": WEB_SEARCH_NODE, "Hallucinated": GENERATE_NODE},
)
workflow.add_edge(WEB_SEARCH_NODE, GENERATE_NODE)
workflow.add_edge(GENERATE_NODE, END)

app = workflow.compile()
app.get_graph().draw_mermaid_png(output_file_path="graph.png")
