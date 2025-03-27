from langchain_core.tools import tool
from langchain_cohere import ChatCohere
from dotenv import load_dotenv
from langchain.schema import Document
from langchain_core.runnables import RunnableConfig
import uuid
from .memory import store,get_user_id
from typing import List
load_dotenv()


@tool
def save_recall_memory(memory: str, config: RunnableConfig) -> str:
    """Save memory to vectorstore for later semantic retrieval."""
    user_id = get_user_id(config)
    document = Document(
        page_content=memory, id=str(uuid.uuid4()), metadata={"user_id": user_id}
    )
    store.add_documents([document])
    return memory


@tool
def search_recall_memories(query: str, config: RunnableConfig) -> List[str]:
    """Search for relevant memories."""
    user_id = get_user_id(config)

    def _filter_function(doc: Document) -> bool:
        return doc.metadata.get("user_id") == user_id

    documents = store.similarity_search(
        query, k=3, filter=_filter_function
    )
    return [document.page_content for document in documents]


@tool
def notify_user(notification: str):
    """Send a notification to the user."""
    return f"Notification sent: {notification}"

@tool
def send_email(to: str, subject: str, body: str):
    """Send an email."""
    return f"Email sent to {to} with subject: {subject}"

tools=[send_email,notify_user,search_recall_memories,save_recall_memory]

llm_model = ChatCohere(
    temperature=0
).bind_tools(
    tools=tools
)

tools_by_name = {
    tool.name:tool
    for tool in tools
}