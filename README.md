# War and Peace Chatbot

The **War and Peace Chatbot** is an interactive RAG-based app built
with LangChain and Streamlit. It allows users to ask questions about 
*War and Peace* and get intelligent, context-aware answers by retrieving 
relevant text chunks from the book and generating responses using an LLM.

---

## Documentation

Full setup and usage instructions are provided below.

---

## Installation

To install and run the chatbot locally, follow these steps:

### Prerequisites

- Python 3.13.2 or higher
- Git
- Virtual environment tool (e.g., `.venv` or `virtualenv`)

---

## Environment Variables

Before running the app, you need to create a `.env` file in the root directory and define the following environment variables:

### Required Variables

| Variable Name      | Type   | Description                                                   |
|--------------------|--------|---------------------------------------------------------------|
| `GROQ_API_KEY`      | string | Your API key from [Groq](https://console.groq.com/keys)           |
| `CHROMA_DB_DIR`     | string | Path where Chroma should store the vector database locally    |


### Steps

1. **Clone the repository**:

   ```sh
   git clone https://github.com/mar-vyn23/rag-with-streamlit.git

1. **Enter the project directory**:

   ```sh
   cd rag-with-streamlit

   ```

1. **Setup your virtual environment**:

   ```sh
   py -m venv .venv

   ```
1. **Activate your virtual environment**:

   ```sh
   .venv\Scripts\activate

   ```
1. Install all the necessary requirements:

   ```sh
   pip install -r requirements.txt

   ```

1. Run your application:

   ```sh
   streamlit run streamlit_ui.py

   ```

1. Run the url below in your favourite browser to check it out:

   ```sh
   http://localhost:8501/

   ```
