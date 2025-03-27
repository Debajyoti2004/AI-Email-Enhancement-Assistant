from langchain_core.vectorstores import InMemoryVectorStore
from langchain_core.runnables import RunnableConfig
from langchain_core.tools import tool
import google.generativeai as genai
from typing import List
from langchain.schema import Document
from langchain_google_genai import GoogleGenerativeAIEmbeddings

genai.configure(api_key="GOOGLE_API_KEY")
embedding_model = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
store = InMemoryVectorStore(
    embedding=embedding_model
)


def get_user_id(config: RunnableConfig) ->str:
    user_id = config["configurable"].get("user_id")
    if user_id is None:
        raise ValueError("User ID needs to be provided to save a memory.")

    return user_id



