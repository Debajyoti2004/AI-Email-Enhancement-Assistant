import sys
import os
import graphviz
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from langgraph.graph import END, StateGraph, START
from build_state import GraphState

from build_state import (
    refinement_grader,
    decide_to_grade,
    query_router,
    refined_email_generate,
    generate_response,
    email_type_finder,
    composing_email,
    decide_to_route,
    save_previous_email,
    document_grader
)

response_rag_agent_workflow = None

def visualize_workflow(workflow, filename="workflow"):
    dot = graphviz.Digraph(format="png")
    
    for node in workflow.nodes:
        dot.node(node, node)

    for src, dst in workflow.edges:
        dot.edge(src, dst)

    dot.render(filename, format="png", cleanup=True)
    print(f"Workflow diagram saved as {filename}.png")

def rag_agent_workflow():
    global response_rag_agent_workflow
    if response_rag_agent_workflow is not None:
        print("Workflow already compiled. Returning existing instance.")
        return response_rag_agent_workflow

    print("Compiling workflow...")
    workflow = StateGraph(GraphState)

    nodes = {
        "Refinement Grader": refinement_grader,
        "Query Router": query_router,
        "Refined Email Generate": refined_email_generate,
        "Generate Response": generate_response,
        "Email Type Finder": email_type_finder,
        "Composing Email": composing_email,
        "Retrieve Pevious  Response":save_previous_email
    }

    for name, func in nodes.items():
        workflow.add_node(name, func)

    workflow.add_edge(START, "Refinement Grader")

    workflow.add_conditional_edges(
        "Refinement Grader",
        decide_to_grade,
        {
            "Refining": "Refined Email Generate",
            "Generate": "Query Router"
        }
    )

    workflow.add_conditional_edges(
        "Query Router",
        decide_to_route,
        {
            "Composing_Emails": "Email Type Finder",
            "Generate_Response": "Retrieve Pevious  Response"
        }

    )
    workflow.add_conditional_edges(
        "Retrieve Pevious  Response",
        document_grader,
        {
            "YES":"Generate Response",
            "NO":"Email Type Finder"
        }
    )
    workflow.add_edge("Email Type Finder","Composing Email")

    workflow.add_edge("Composing Email", END)
    workflow.add_edge("Generate Response", END)
    workflow.add_edge("Refined Email Generate", END)

    response_rag_agent_workflow = workflow.compile()
    return response_rag_agent_workflow
