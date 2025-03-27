import sys
import os
import graphviz
from langgraph.graph import END, StateGraph, START
from build_state import GraphState, triage_router, decide_to_triage
from response_agent import rag_agent_workflow
from notify_agent import compile_notify_agent

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

response_rag_agent = rag_agent_workflow()
notify_rag_agent = compile_notify_agent()
main_compile_workflow = None

def rag_agent_state(state: GraphState) -> GraphState:
    return response_rag_agent.invoke(state)

def notify_agent_state(state: GraphState) -> GraphState:
    agent_state = {
        **state,
        "recall_memories": [],
        "messages": []
    }
    updated_agent_state = notify_rag_agent.invoke(agent_state)

    return {
        key: updated_agent_state[key] for key in GraphState.__annotations__
    }

def email_assistant_workflow():
    global main_compile_workflow
    if main_compile_workflow is not None:
        print("Workflow already compiled. Returning existing instance.")
        return main_compile_workflow

    print("Compiling workflow...")
    workflow = StateGraph(GraphState)
    workflow.add_node("Response RAG_Agent", rag_agent_state)
    workflow.add_node("Triage_Router", triage_router)
    workflow.add_node("Notify Rag Agent", notify_agent_state)
    
    workflow.add_edge(START, "Triage_Router")
    workflow.add_conditional_edges(
        "Triage_Router",
        decide_to_triage,
        {
            "response-agent": "Response RAG_Agent",
            "notify": "Notify Rag Agent",
            "end": END
        }
    )
    
    main_compile_workflow = workflow.compile()
    return main_compile_workflow
