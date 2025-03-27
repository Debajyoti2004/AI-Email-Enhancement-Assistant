import json
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain.vectorstores import FAISS
from langchain.schema import Document
from dotenv import load_dotenv
import os

load_dotenv()
documents = [
    Document(page_content="""From: manager@example.com
                            To: team@example.com
                            Subject: Meeting Reminder
                            Body: Reminder for the 3 PM meeting.
                            """, metadata={"Subject": "Meeting Reminder", "user_id": "user_123"})
]
embedding_model = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
fiass_index_path = os.getenv("FAISS_INDEX_PATH")

if os.path.exists(fiass_index_path):
    vector_store = FAISS.load_local(fiass_index_path, embedding_model,allow_dangerous_deserialization=True)
    print("Loaded existing FAISS index......")
else:
    vector_store = FAISS.from_documents(documents, embedding_model)
    vector_store.save_local(fiass_index_path) 
    print(" Created new FAISS index......")