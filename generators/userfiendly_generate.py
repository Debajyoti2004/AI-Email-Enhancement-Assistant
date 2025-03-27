from langchain_ollama import ChatOllama
from langchain_core.output_parsers import JsonOutputParser
from langchain_core.prompts import PromptTemplate

prompt = PromptTemplate(
    template="""
    You are an expert email assistant dedicated to refining and improving email responses. 

    **Original Email:**  
    {email}  

    **List of Improvements Needed:**  
    {ImprovementList}  

    ### Task:  
    Improve the given email while maintaining professionalism and clarity.  
    Ensure the output is returned in **strict JSON format** as follows:  

    ```json
    {{
        "From": "<original sender>",
        "To": "<recipient>",
        "Subject": "<refined subject>",
        "Body": "<improved email body>"
    }}
    ```  

    **Rules:**  
    - Keep the response formal and professional.  
    - Preserve the original intent while enhancing grammar, clarity, and tone.  
    - Ensure the email structure remains intact.  
    - Return only the JSON output without any extra text.  
    """,
    input_variables=["email", "ImprovementList"]
)

llm = ChatOllama(
    model="llama3.2-vision", 
    temperature=0
)

email_refiner = (
    prompt 
    | llm 
    | JsonOutputParser()
)
