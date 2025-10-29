from dotenv import load_dotenv
load_dotenv()
from langgraph.graph import StateGraph, START, END
from graph.consts import RETRIEVE_NODE, WEB_SEARCH_NODE, GRADE_DOCUMENTS_NODE, GENERATE_NODE
from graph.nodes import retrieve, web_search, grade_documents, generate
from graph.state import GraphState


def decide_to_generate(state: GraphState) -> StateGraph:
    if state["web_search"]:
        return WEB_SEARCH_NODE
    else:
        return GENERATE_NODE

workflow = StateGraph(GraphState)
workflow.add_node(RETRIEVE_NODE, retrieve)
workflow.add_node(WEB_SEARCH_NODE, web_search)
workflow.add_node(GRADE_DOCUMENTS_NODE, grade_documents)
workflow.add_node(GENERATE_NODE, generate)

workflow.add_edge(START, RETRIEVE_NODE)
workflow.add_edge(RETRIEVE_NODE, GRADE_DOCUMENTS_NODE)
workflow.add_conditional_edges(GRADE_DOCUMENTS_NODE, decide_to_generate, {WEB_SEARCH_NODE: WEB_SEARCH_NODE, GENERATE_NODE: GENERATE_NODE})
workflow.add_edge(WEB_SEARCH_NODE, GENERATE_NODE)
workflow.add_edge(GENERATE_NODE, END)

app = workflow.compile()
app.get_graph().draw_mermaid_png(output_file_path="graph.png")