from langchain_cohere import ChatCohere
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import JsonOutputParser
from pydantic import BaseModel, Field
from dotenv import load_dotenv
from typing_extensions import Literal

load_dotenv()

class EmailClassifier(BaseModel):
    email_type: Literal[
        "Technical",
        "Formal",
        "Informal",
        "Marketing",
        "Customer Support",
        "HR & Recruitment",
        "Finance & Billing",
        "Legal",
        "Personal"
    ] = Field(
        ..., description="Classifies the email into a specific category."
    )

prompt = PromptTemplate(
    template="""
    You are an expert email classification assistant. Your task is to classify the given email into one of the following categories:

    - **Technical**: Emails related to technical discussions, troubleshooting, software development, or engineering topics.
    - **Formal**: Professional, business-related, or official emails with a structured format.
    - **Informal**: Casual, friendly, or non-business-related emails.
    - **Marketing**: Promotional emails, newsletters, or advertisements.
    - **Customer Support**: Emails related to support requests, issue resolution, or customer service inquiries.
    - **HR & Recruitment**: Emails regarding job applications, hiring, or HR-related matters.
    - **Finance & Billing**: Emails related to invoices, transactions, or financial statements.
    - **Legal**: Emails concerning contracts, policies, or legal matters.
    - **Personal**: Non-work-related personal emails.

    **Email Content:**
    {query}

    **Classification Task:**
    Determine the appropriate category and return only the category name as output.
    """,
    input_variables=["query"]
)

llm = ChatCohere(
    temperature=0
    ).bind_tools(
        tools=[EmailClassifier]
    )

email_type_router_chain = (
    prompt
    | llm
)


