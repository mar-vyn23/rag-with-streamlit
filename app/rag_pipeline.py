from langchain.chains import RetrievalQA
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
from langchain_groq import ChatGroq
from pathlib import Path
import os

from dotenv import load_dotenv
load_dotenv()

# Load and split documents
def load_vectorstore():
    #load the document
    with open("data/war_and_peace.txt", "r", encoding="utf-8") as file:
        document = file.read()

    #split the document
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size = 250,
        chunk_overlap = 100,
    )

    texts = text_splitter.create_documents([document])

    #semantic meaning
    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-mpnet-base-v2",
        model_kwargs = {'device':'cpu'}
        )
    
    #database
    vectorstore = Chroma(
        collection_name = "collection_name", 
        embedding_function=embeddings,
        persist_directory="chroma_db",
    )

    return vectorstore

retriever = load_vectorstore().as_retriever()

#Groq LLM
llm = ChatGroq(
    api_key=os.getenv("GROQ_API_KEY"),
    model_name="llama-3.1-8b-instant",
    temperature=0.8,
    max_tokens=200,
)

#building the chain
qa_chain = RetrievalQA.from_chain_type(
    chain_type="stuff",
    llm=llm,
    retriever=retriever,
)

#function that queries user questions
def ask_question(query):
    return qa_chain.run(query)
