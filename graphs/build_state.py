from generators import (
    email_generator,
    email_refiner,
    email_reply_chain
)
from graders import (
    refinementGrader_chain
)
from routers import (
    email_type_router_chain,
    query_classifier_chain,
    triage_router_chain
)
from memory_store import (
    store_email_response,retrieve_email_responses
)
from typing_extensions import TypedDict
from langgraph.types import Command
from typing import Literal
from langgraph.graph import END
import json

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



def triage_router(state: GraphState):
    email = state["email_input"]
    result = triage_router_chain.invoke({"email":email})
    response_str = result.additional_kwargs["tool_calls"][0]["function"]["arguments"]

    response_dict = json.loads(response_str)
    response = response_dict["classification"]
    return {
        **state,
        "classification":response
    }

def decide_to_triage(state):
    response = state["classification"]
    if response == "Respond":
        print(" Classification: RESPOND - This email requires a response")
        return "response-agent"
    elif response == "Ignore":
        print(" Classification: IGNORE - This email can be safely ignored")
        return "end"
    elif response == "Notify":
        print(" Classification: NOTIFY - This email contains important information")
        return "notify"
    else:
        raise ValueError(f"Invalid classification: {response}")
    

def refinement_grader(state):
    query = state["query"]
    email = state["email_input"]

    response = refinementGrader_chain.invoke({"query":query,"email":email})
    response_str = response.additional_kwargs["tool_calls"][0]["function"]["arguments"]
    response_dict = json.loads(response_str)

    needing_improvement = response_dict.get("needingImprovement","no")
    improvement_list = response_dict.get("ImprovementList","")
    return {
        **state,
        "needing_improvement":needing_improvement,
        "improvement_list":improvement_list
    }

def decide_to_grade(state):
    if state["needing_improvement"]=="yes":
        return "Refining"
    else:
        return "Generate"
    
def query_router(state):
    query = state["query"]
    response = query_classifier_chain.invoke({"query":query})
    result = response.additional_kwargs["tool_calls"][0]["function"]["name"]
    
    return {
        **state,
        "sub_router_state":result
    }

def decide_to_route(state):
    result = state["sub_router_state"]
    
    if result == "ComposingEmails":
        return "Composing_Emails"
    elif result == "GenerateResponse":
        return "Generate_Response"
    
def refined_email_generate(state):
    email = state["email_input"]
    improvement_list = state["improvement_list"]

    response = email_refiner.invoke({"email":email,"ImprovementList":improvement_list})
    return {
        **state,
         "generation":response
    }

def document_grader(state):
    previous_email = state["previous_email"]

    if not previous_email:
        return "NO"
    else:
        return "YES"
    
def save_previous_email(state):
    user_id = state["user_id"]
    email_subject = state["email_input"]["Subject"]
    previous_email = retrieve_email_responses(user_id=user_id,subject=email_subject)

    return {
        **state,
        "previous_email":previous_email
    }
def generate_response(state):
    user_id = state["user_id"]
    previous_email = state["previous_email"]
    current_email = state["email_input"]
    store_email_response(user_id=user_id,email=current_email)

    response = email_reply_chain.invoke({"previous_response":previous_email,"current_email":current_email})
    
    store_email_response(user_id=user_id,email=response)
    return {
        **state,
        "generation":response
    }

def email_type_finder(state):
    query = state["query"]
    response = email_type_router_chain.invoke({"query":query})

    result = json.loads(response.additional_kwargs["tool_calls"][0]["function"]["arguments"])
    return {
        **state,
        "email_type":result["email_type"]
    }

def composing_email(state):
    email_type=state["email_type"]
    email = state["email_input"]
    query = state["query"]
    user_id = state["user_id"]

    response = email_generator.invoke({"query":query,
                                       "email_type":email_type,
                                       "email":email})
    
    store_email_response(user_id=user_id,email=response)
    return {
        **state,
        "generation":response
    }








