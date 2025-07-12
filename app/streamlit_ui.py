import streamlit as st
from rag_pipeline import ask_question

st.set_page_config(page_title="War and Peace Chatbot", layout="wide")
st.title("Chat About *War and Peace*")

# Sidebar instructions
with st.sidebar:
    st.markdown("INSTRUCTIONS")
    st.write("Ask any question about the book; *War and Peace*.")
    st.write("This assistant will retrieve context from the book and answer you.")
    if st.button("Clear Conversation"):
        st.session_state.conversation = []
        st.rerun()

# Initialize conversation state
if "conversation" not in st.session_state:
    st.session_state.conversation = []

# Display chat history using chat_message
for turn in st.session_state.conversation:
    with st.chat_message("user"):
        st.markdown(turn["question"])
    with st.chat_message("assistant"):
        st.markdown(turn["answer"])

# Input box for new message
user_input = st.chat_input("Ask your next question here...")

if user_input:
    with st.chat_message("user"):
        st.markdown(user_input)

    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            response = ask_question(user_input)
        st.markdown(response)

    # Store turn in conversation history
    st.session_state.conversation.append({
        "question": user_input,
        "answer": response
    })
