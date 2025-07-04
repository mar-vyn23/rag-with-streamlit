from langchain.chains import RetrievalQA
from langchain_community.document_loaders import DirectoryLoader
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain.text_splitter import CharacterTextSplitter
from langchain_groq import ChatGroq
import os

from dotenv import load_dotenv
load_dotenv()

# Load and split documents
def load_vectorstore():
    loader = DirectoryLoader("data", glob="**/*.txt")
    docs = loader.load()

    text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=20)
    texts = text_splitter.split_documents(docs)

    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    vectorstore = FAISS.from_documents(texts, embeddings)
    return vectorstore

retriever = load_vectorstore().as_retriever()

#Groq LLM
llm = ChatGroq(
    api_key=os.getenv("GROQ_API_KEY"),
    model_name="llama-3.1-8b-instant",
    temperature=0.7,
    # max_tokens=50,
)

qa_chain = RetrievalQA.from_chain_type(
    chain_type="stuff",
    llm=llm,
    retriever=retriever,
)

def ask_question(query):
    return qa_chain.run(query)
