from fastapi import FastAPI, HTTPException, UploadFile, File, BackgroundTasks
from pydantic import BaseModel
from langchain.text_splitter import CharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain_groq import ChatGroq
from langchain.schema import HumanMessage, SystemMessage
from dotenv import load_dotenv
from fastapi.middleware.cors import CORSMiddleware
import os
from pinecone import Pinecone, ServerlessSpec
from langchain_pinecone import PineconeVectorStore
from functools import lru_cache
import asyncio
from typing import List, Dict

app = FastAPI()
load_dotenv()

app.add_middleware(
   CORSMiddleware,
   allow_origins=["*"],
   allow_credentials=True,
   allow_methods=["*"],
   allow_headers=["*"],
)

PINECONE_INDEX_NAME = "documents-index"
BATCH_SIZE = 100
MAX_RETRIES = 3
RETRY_DELAY = 1

class Query(BaseModel):
   question: str

@lru_cache(maxsize=1)
def get_pinecone_client():
   return Pinecone(
       api_key=os.getenv("PINECONE_API_KEY"),
       pool_threads=30
   )

@lru_cache(maxsize=1)
def get_pinecone_index():
   pc = get_pinecone_client()
   if PINECONE_INDEX_NAME not in pc.list_indexes().names():
       pc.create_index(
           name=PINECONE_INDEX_NAME,
           dimension=1536,
           metric="cosine",
           spec=ServerlessSpec(
               cloud='aws',
               region='us-east-1'
           )
       )
   return pc.Index(PINECONE_INDEX_NAME)

@lru_cache(maxsize=1)
def get_embeddings():
   return OpenAIEmbeddings()

@lru_cache(maxsize=1)
def get_chat_model():
   return ChatGroq(model="llama-3.3-70b-versatile", api_key=os.getenv("GROQ_API_KEY"))

async def process_batch(batch: List[Dict], embeddings: OpenAIEmbeddings, index, retry_count=0):
   try:
       texts = [doc["content"] for doc in batch]
       embedding_vectors = embeddings.embed_documents(texts)
       
       vectors = [
           {
               "id": str(hash(doc["content"])),
               "values": vector,
               "metadata": {
                   "text": doc["content"],
                   **doc["metadata"]
               }
           }
           for doc, vector in zip(batch, embedding_vectors)
       ]
       
       index.upsert(vectors=vectors)
   except Exception as e:
       if retry_count < MAX_RETRIES:
           await asyncio.sleep(RETRY_DELAY)
           await process_batch(batch, embeddings, index, retry_count + 1)
       else:
           raise e

async def process_document(content: bytes, filename: str):
   try:
       text = content.decode('utf-8')
       text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=0)
       texts = text_splitter.split_text(text)
       documents = [{"content": chunk, "metadata": {"source": filename}} for chunk in texts]
       
       embeddings = get_embeddings()
       index = get_pinecone_index()
       
       batches = [documents[i:i + BATCH_SIZE] for i in range(0, len(documents), BATCH_SIZE)]
       tasks = [process_batch(batch, embeddings, index) for batch in batches]
       
       await asyncio.gather(*tasks)
       
   except Exception as e:
       print(f"Error processing document: {str(e)}")

@app.post("/upload")
async def upload_document(file: UploadFile = File(...)):
   if not file.filename.endswith('.txt'):
       raise HTTPException(400, "Only .txt files are supported")
   
   try:
       content = await file.read()
       text = content.decode('utf-8')
       text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=0)
       texts = text_splitter.split_text(text)
       documents = [{"content": chunk, "metadata": {"source": file.filename}} for chunk in texts]
       
       embeddings = get_embeddings()
       index = get_pinecone_index()
       
       batches = [documents[i:i + BATCH_SIZE] for i in range(0, len(documents), BATCH_SIZE)]
       tasks = [process_batch(batch, embeddings, index) for batch in batches]
       
       await asyncio.gather(*tasks)
       
       return {"message": "File processed successfully"}
   except Exception as e:
       raise HTTPException(status_code=500, detail=f"Error processing file: {str(e)}")
   
   
def get_vectorstore(embeddings):
   index = get_pinecone_index()
   return PineconeVectorStore(
       index=index,
       embedding=embeddings,
       text_key="text"
   )

@app.post("/query")
async def query_documents(query: Query):
   try:
       embeddings = get_embeddings()
       vectorstore = get_vectorstore(embeddings)
       relevant_docs = vectorstore.similarity_search(query.question, k=2)
       
       combined_input = (
           f"Question: {query.question}\n\n"
           f"Relevant Documents:\n"
           + "\n\n".join([doc.page_content for doc in relevant_docs])
           + "\n\nPlease provide an answer based only on the provided documents. "
           "If the answer is not found in the documents, respond with 'I'm not sure'."
       )
       
       model = get_chat_model()
       messages = [
           SystemMessage(content="You are a helpful assistant."),
           HumanMessage(content=combined_input)
       ]
       
       result = model(messages)
       
       return {
           "answer": result.content,
           "sources": [{"content": doc.page_content, "source": doc.metadata.get("source")} 
                      for doc in relevant_docs]
       }
   except Exception as e:
       raise HTTPException(status_code=500, detail=f"Error querying documents: {str(e)}")

@app.delete("/delete-all")
async def delete_all():
   try:
       index = get_pinecone_index()
       index.delete(delete_all=True)
       return {"message": "All vectors deleted from Pinecone"}
   except Exception as e:
       raise HTTPException(status_code=500, detail=f"Error: {str(e)}")

if __name__ == "__main__":
   import uvicorn
   uvicorn.run(app, host="0.0.0.0", port=8000)
