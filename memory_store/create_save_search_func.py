from .create_memory_store import vector_store
from langchain.schema import Document
import json
import os

def store_email_response(user_id: str, email: dict):
    subject = email["Subject"]
    body_str = json.dumps(email)

    doc = Document(page_content=body_str, metadata={"user_id": user_id, "subject": subject})
    vector_store.add_documents([doc])
    vector_store.save_local(os.getenv("FAISS_INDEX_PATH")) 
    print(f"Email response for '{subject}' (User: {user_id}) saved.")


def retrieve_email_responses(user_id: str, subject: str):
    results = vector_store.similarity_search(subject, k=5)
    
    emails = [
        res.page_content
        for res in results
        if res.metadata.get("user_id") == user_id and res.metadata.get("subject") == subject
    ]
    
    return emails if emails else None  