import json
from .create_tools import tools_by_name, llm_model, search_recall_memories
from typing import Annotated, Sequence, TypedDict, List
from langchain_core.messages import BaseMessage, ToolMessage, SystemMessage
from langchain_core.runnables import RunnableConfig
from langgraph.graph.message import add_messages
from .create_prompt import prompt
from langchain_core.messages import get_buffer_string
import tiktoken


tokenizer = tiktoken.get_encoding("cl100k_base")

class GraphState(TypedDict):
    email_input:dict
    email_type:str
    query:str
    previous_email:list[str]
    user_id:str
    generation:dict
    sub_router_state:str
    improvement_list:list[str]
    needing_improvement:str
    classification:str


class AgentState(GraphState):
    """The state of the agent."""
    recall_memories: List[str]
    messages: Annotated[Sequence[BaseMessage], add_messages]


def tool_node(state: AgentState):
    outputs = []
    for tool_call in state["messages"][-1].tool_calls:
        tool_result = tools_by_name[tool_call["name"]].invoke(tool_call["args"])
        outputs.append(
            ToolMessage(
                content=json.dumps(tool_result),
                name=tool_call["name"],
                tool_call_id=tool_call["id"],
            )
        )
    return {"messages": outputs}


def agent(state: AgentState) -> AgentState:
    """Process the current state and generate a response using the LLM."""
    bound = prompt | llm_model
    recall_str = (
        "<recall_memory>\n" + "\n".join(state["recall_memories"]) + "\n</recall_memory>"
    )
    prediction = bound.invoke(
        {
            "messages": state["messages"],
            "recall_memories": recall_str,
        }
    )
    return {"messages": [prediction]}


def load_memories(state: AgentState, config: RunnableConfig) -> AgentState:
    """Load relevant memories for the conversation."""
    convo_str = get_buffer_string(state["messages"])
    convo_str = tokenizer.decode(tokenizer.encode(convo_str)[:2048])
    recall_memories = search_recall_memories.invoke(query=convo_str, config=config)
    return {"recall_memories": recall_memories}


def route_tools(state: AgentState):
    """Determine the next action: tool execution or response finalization."""
    last_message = state["messages"][-1]
    return "tools" if last_message.tool_calls else "end"
