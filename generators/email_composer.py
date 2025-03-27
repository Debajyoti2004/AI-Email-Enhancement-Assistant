from langchain_ollama import ChatOllama
from langchain_core.prompts import PromptTemplate

prompt = PromptTemplate(
    template="""
    You are a **{email_type} Email Assistant**. Your task is to generate a well-structured and professional email body based on the provided details.

    **Email Type:** {email_type}

    **Email Details:**
    {email}

    **User Query:**
    {query}

    ### Task:
    - Compose a natural and professional email body relevant to the subject and query.
    - Ensure clarity, conciseness, and accuracy.
    - Adapt tone and content based on the specified email type.

    **Return only JSON output** in the following format:

    ```json
    {{
        "From": "<from> collect from email details part above",
        "To": "<to> collect from email details part above",
        "Subject": "<subject> collect from email details part above",
        "Body": "<Generated email body>"
    }}
    ```

    **Rules:**
    - Maintain a professional tone suitable for the given email type.
    - Avoid predefined templates; generate a fresh response based on context.
    - Ensure the response is well-structured and relevant.
    """,
    input_variables=["email_type", "email", "query"]
)

llm = ChatOllama(
    model="llama3.2-vision",
    temperature=0
)

email_generator = (
    prompt 
    | llm
)


