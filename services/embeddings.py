from langchain_nvidia_ai_endpoints import ChatNVIDIA, NVIDIAEmbeddings
from langchain_community.document_loaders import PyPDFDirectoryLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_community.vectorstores import FAISS
from configs import client
import os
def vectorEmbeddings():
    embeddings = NVIDIAEmbeddings()
    faiss_index_path = "./faiss_index"
    
    # Check if a saved FAISS index exists
    if os.path.exists(faiss_index_path):
        print("Loading existing embeddings from disk...\n")
        vectors = FAISS.load_local(faiss_index_path, embeddings, allow_dangerous_deserialization=True)
        print("Embeddings loaded from disk.\n")
    else:
        print("No saved embeddings found. Generating new embeddings (this may take a while)...\n")
        loader = PyPDFDirectoryLoader("./books")
        docs = loader.load()
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=700, chunk_overlap=50)
        final_documents = text_splitter.split_documents(docs)
        vectors = FAISS.from_documents(final_documents, embeddings)
        vectors.save_local(faiss_index_path)  # Save to disk
        print("Embeddings generated and saved to disk.\n")
    return vectors