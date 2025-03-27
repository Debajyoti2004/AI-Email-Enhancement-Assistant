import json
from langchain_ollama import ChatOllama
from langchain_core.output_parsers import JsonOutputParser
from langchain_core.prompts import PromptTemplate

prompt = PromptTemplate(
    template="""
    You are an expert assistant in analyzing email conversations and generating professional replies.
    
    Previous Email Response:
    {previous_response}
    
    Current Email:
    {current_email}
    
    Based on this conversation, generate a professional reply in JSON format:
    {{
        "From": "assistant@example.com",
        "To": "",
        "Subject": "",
        "Body": ""
    }}
    """,
    input_variables=["previous_response", "current_email"]
)

ollama_llm = ChatOllama(
    model="llama3.2-vision", 
    temperature=0
)

email_reply_chain = (
    prompt 
    | ollama_llm 
    | JsonOutputParser()
)

