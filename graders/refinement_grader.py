from langchain_cohere import ChatCohere
from langchain_core.prompts import PromptTemplate
from pydantic import BaseModel, Field
from dotenv import load_dotenv

load_dotenv()

class RefinementGrader(BaseModel):
    needingImprovement: str = Field(
        ..., description=""""Return 'yes' if user,s query asking about refining or correcting it, otherwise return "no"."""
    )
    ImprovementList: list[str] = Field(
        ..., description="List specific areas needing improvement, such as: "
                        "- Tone Improvement "
                        "- Grammar Improvement "
                        "- Clarity Improvement "
                        "- Too vague "
                        "- Lacks professionalism "
                        "- Missing subject line "
                        "- No introduction or closing."
    )

refinement_prompt = PromptTemplate(
    template="""
        You are an expert in email communication analysis. 
        
        - If the user's query explicitly asks for **refining, correcting, or improving** their email, analyze the given email.
        - If the query does **not** ask for improvement, return "No refinement needed."

        **User Query:** {query}

        **Email to Review:**  
        {email}

        If refinement is required, output:
        - **needingImprovement**: "yes" or "no"
        - **ImprovementList**: List of suggested improvements.
    """,
    input_variables=["query", "email"]
)

structured_llm = ChatCohere(
    temperature=0
).bind_tools(
    tools=[RefinementGrader]
)

refinementGrader_chain = (
    refinement_prompt 
    | structured_llm
)
