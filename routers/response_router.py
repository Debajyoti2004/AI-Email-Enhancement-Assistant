from langchain_cohere import ChatCohere
from langchain_core.prompts import PromptTemplate
from pydantic import BaseModel, Field
from dotenv import load_dotenv

load_dotenv()

class ComposingEmails(BaseModel):
    composing_emails: str = Field(
        ..., description="Select this route if the user's query pertains to drafting, composing, or structuring an email, including formal and informal correspondence."
    )

class GenerateResponse(BaseModel):
    generate_response: str = Field(
        ..., description="Select this route if the user is specifically asking what to reply to an email they received."
    )

query_classifier_prompt = PromptTemplate(
    template="""
        You are an expert classifier that selects the appropriate tool for the user's query:

        - **ComposingEmails** → Use this when the user wants to draft, compose, or structure an email (formal or informal).
        - **GenerateResponse** → Use this **only if the user is explicitly asking what to reply to an email they received**.

        **User Query**:
        {query}

        Which tool will you use?
    """,
    input_variables=["query"]
)

structured_llm = ChatCohere(
    temperature=0
).bind_tools(
    tools=[ComposingEmails, GenerateResponse]
)

query_classifier_chain = (
    query_classifier_prompt 
    | structured_llm
)
