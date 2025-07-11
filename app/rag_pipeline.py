from langchain.chains import RetrievalQA
#from langchain_community.document_loaders import TextLoader
#from langchain_community.vectorstores import FAISS
#from langchain_community.embeddings import HuggingFaceEmbeddings
#from langchain.text_splitter import CharacterTextSplitter
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
    # loader = DirectoryLoader("data", glob="**/*.txt")
    # docs = loader.load()
    with open("data/war_and_peace.txt", "r", encoding="utf-8") as file:
        document = file.read()

    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size = 250,
        chunk_overlap = 50,
    )

    texts = text_splitter.create_documents([document])

    # text_splitter = CharacterTextSplitter(
    #     chunk_size=1000, 
    #     chunk_overlap=20
    #     )
    # texts = text_splitter.split_documents(docs)

    # text_splitter = CharacterTextSplitter.from_tiktoken_encoder(
    #     chunk_size =100,
    #     chunk_overlap=20
    # )
    # texts = text_splitter.split_documents(docs)

    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-mpnet-base-v2",
        model_kwargs = {'device':'cpu'}
        )
    
    # vectorstore = FAISS.from_documents(texts, embeddings)
    # return vectorstore

    # vectorstore = Chroma.from_documents(
    #     texts,
    #     embeddings = embeddings,
    #     persist_directory = "chroma_db"

    #for reloading the chroma db instead of rebuilding it
    vectorstore = Chroma(
        persist_directory="chroma_db", 
        embedding_function=embeddings
    )

    return vectorstore

retriever = load_vectorstore().as_retriever()

#Groq LLM
llm = ChatGroq(
    api_key=os.getenv("GROQ_API_KEY"),
    model_name="llama-3.1-8b-instant",
    temperature=0.8,
    # max_tokens=50,
)

qa_chain = RetrievalQA.from_chain_type(
    chain_type="stuff",
    llm=llm,
    retriever=retriever,
)

def ask_question(query):
    return qa_chain.run(query)
