from langgraph.graph import START, StateGraph, END
from .create_node import AgentState, load_memories, agent, route_tools,tool_node
from .create_tools import tools
from langgraph.checkpoint.memory import MemorySaver

def compile_notify_agent():
    builder = StateGraph(AgentState)
    builder.add_node(load_memories)
    builder.add_node(agent)
    builder.add_node("tools", tool_node)

    # Graph structure
    builder.add_edge(START, "load_memories")
    builder.add_edge("load_memories", "agent")
    builder.add_conditional_edges("agent", route_tools, ["tools", END])
    builder.add_edge("tools", "agent")

    # Compile with memory checkpointing
    memory = MemorySaver()
    return builder.compile(checkpointer=memory)
