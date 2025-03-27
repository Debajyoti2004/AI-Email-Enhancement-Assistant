from langchain_cohere import ChatCohere
from langchain_core.prompts import PromptTemplate
from pydantic import BaseModel, Field
from dotenv import load_dotenv
from typing_extensions import Literal

load_dotenv()

class TriageRouter(BaseModel):
    reasoning: str = Field(
        description="Step-by-step reasoning behind the classification."
    )
    classification: Literal["Ignore", "Respond", "Notify"] = Field(
        description="The classification of an email: 'ignore' for irrelevant emails, "
                    "'notify' for important information that doesn't need a response, "
                    "'respond' for emails that need a reply."
    )

triage_router_prompt = PromptTemplate(
    template="""
        You are an expert email classifier that categorizes emails into three types:
        
        1. **Ignore** → Irrelevant, spam, or requires no action.
        2. **Notify** → Important information but does not need a response.
        3. **Respond** → Emails that need a reply.

        **Analyze the email content and provide a classification along with reasoning.**
        
        **Email Content**:
        {email}

        **Response Format**:
        Reasoning: <Step-by-step explanation>
        Classification: <Ignore / Respond / Notify>
    """,
    input_variables=["email"]
)

structured_llm = ChatCohere(
    temperature=0
).bind_tools(
    tools=[TriageRouter]
)

triage_router_chain = (
    triage_router_prompt 
    | structured_llm
)
