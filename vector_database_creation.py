import os
from uuid import uuid4
from langchain.text_splitter import CharacterTextSplitter
from langchain.document_loaders import TextLoader
from langchain.vectorstores import Chroma
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain_huggingface import HuggingFaceEmbeddings
from dotenv import load_dotenv

load_dotenv()

current_dir = os.path.dirname(os.path.abspath(__file__))
file_path = os.path.join(current_dir, "documents", "lord_of_the_rings.txt")
persistent_directory = os.path.join(current_dir, "db", "chroma_db")

if not os.path.exists(persistent_directory):
    print("Persistent directory does not exist. Initializing vector store...")

    if not os.path.exists(file_path):
        raise FileNotFoundError(
        )

    loader = TextLoader(file_path)
    documents = loader.load()

    text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=50)
    docs = text_splitter.split_documents(documents=documents)

    for i, doc in enumerate(docs):
        doc.metadata["id"] = str(uuid4())  
    print("\n--- Document Chunks Information ---")
    print(f"Number of document chunks: {len(docs)}")
    print(f"Sample Chunk: \n {docs[0].page_content}\n")

    print("\n----- Creating Embeddings -----")
    # embeddings = OpenAIEmbeddings(api_key=os.getenv("OPENAI_API_KEY"))
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

    print("\n----- Creating Vector Store -----")
    db = Chroma.from_documents(
        docs, 
        embeddings, 
        persist_directory=persistent_directory
    )

    db.persist()
    print("Vector store created and persisted successfully.")
else:
    print("Vector store already exists. No need to initialize.")
    #embeddings = OpenAIEmbeddings(api_key=os.getenv("OPENAI_API_KEY"))
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    db = Chroma(
        embedding_function=embeddings, 
        persist_directory=persistent_directory
    )
    print("Loaded existing vector store.")
