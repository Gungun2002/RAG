from fastapi import FastAPI, HTTPException, UploadFile, File
from pydantic import BaseModel
import os
import shutil
from typing import List
from langchain.text_splitter import CharacterTextSplitter
from langchain_community.document_loaders import TextLoader
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_groq import ChatGroq
from langchain_core.messages import HumanMessage, SystemMessage
from dotenv import load_dotenv
from fastapi.middleware.cors import CORSMiddleware

app = FastAPI()
load_dotenv()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# Configuration
BOOKS_DIR = "documents"
DB_DIR = "db"
PERSISTENT_DIR = os.path.join(DB_DIR, "chroma_db_with_metadata")
EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"

# Initialize directories
os.makedirs(BOOKS_DIR, exist_ok=True)
os.makedirs(DB_DIR, exist_ok=True)

class Query(BaseModel):
    question: str

def initialize_vectorstore():
    if not os.path.exists(PERSISTENT_DIR):
        book_files = [f for f in os.listdir(BOOKS_DIR) if f.endswith(".txt")]
        
        documents = []
        for book_file in book_files:
            file_path = os.path.join(BOOKS_DIR, book_file)
            loader = TextLoader(file_path)
            book_docs = loader.load()
            for doc in book_docs:
                doc.metadata = {"source": book_file}
                documents.append(doc)

        text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=0)
        docs = text_splitter.split_documents(documents)
        
        embeddings = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL)
        db = Chroma.from_documents(docs, embeddings, persist_directory=PERSISTENT_DIR)
        return db
    return None

@app.post("/upload")
async def upload_document(file: UploadFile = File(...)):
    if not file.filename.endswith('.txt'):
        raise HTTPException(400, "Only .txt files are supported")
    
    file_path = os.path.join(BOOKS_DIR, file.filename)

    # Save the uploaded file to the documents directory
    with open(file_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)

    # Load the existing vector store or create a new one
    embeddings = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL)

    if os.path.exists(PERSISTENT_DIR):
        db = Chroma(persist_directory=PERSISTENT_DIR, embedding_function=embeddings)
    else:
        os.makedirs(DB_DIR, exist_ok=True)
        db = Chroma(embedding_function=embeddings, persist_directory=PERSISTENT_DIR)

    # Load and process the new document
    loader = TextLoader(file_path)
    new_docs = loader.load()

    # Add metadata and split text
    for doc in new_docs:
        doc.metadata = {"source": file.filename}
    
    text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=0)
    split_docs = text_splitter.split_documents(new_docs)

    # Add new documents to Chroma DB
    db.add_documents(split_docs)  # No need to call db.persist()

    return {"message": "File uploaded and added to vector store successfully"}



@app.post("/query")
async def query_documents(query: Query):
    embeddings = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL)
    db = Chroma(persist_directory=PERSISTENT_DIR, embedding_function=embeddings)
    
    retriever = db.as_retriever(
        search_type="similarity",
        search_kwargs={"k": 3}
    )
    relevant_docs = retriever.invoke(query.question)
    
    combined_input = (
        f"Question: {query.question}\n\n"
        f"Relevant Documents:\n"
        + "\n\n".join([doc.page_content for doc in relevant_docs])
        + "\n\nPlease provide an answer based only on the provided documents. If the answer is not found in the documents, respond with 'I'm not sure'."
    )
    
    model = ChatGroq(
        model="llama-3.3-70b-versatile", 
        api_key=os.getenv("GROQ_API_KEY")
    )
    
    messages = [
        SystemMessage(content="You are a helpful assistant."),
        HumanMessage(content=combined_input)
    ]
    
    result = model.invoke(messages)
    
    return {
        "answer": result.content,
        "sources": [{"content": doc.page_content, "source": doc.metadata.get("source")} 
                   for doc in relevant_docs]
    }

@app.delete("/delete-all")
async def delete_all_files():
    try:
        # Delete all files in the documents directory
        if os.path.exists(BOOKS_DIR):
            shutil.rmtree(BOOKS_DIR)
            os.makedirs(BOOKS_DIR, exist_ok=True)
        
        # Delete all files in the db directory
        if os.path.exists(DB_DIR):
            shutil.rmtree(DB_DIR)
            os.makedirs(DB_DIR, exist_ok=True)
        
        return {"message": "All files in 'documents' and 'db' directories have been deleted."}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error deleting files: {str(e)}")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
