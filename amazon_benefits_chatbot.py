from langchain_community.embeddings import OpenAIEmbeddings
from langchain_community.chat_models import ChatOpenAI
from langchain.chains import ConversationalRetrievalChain
from langchain_community.vectorstores import FAISS
import os
import streamlit as st

def query(question, chat_history):
    # openai_api_key = os.environ.get("OPEN_AI_KEY")
    embeddings = OpenAIEmbeddings()
    new_db = FAISS.load_local("faiss_amazon_index", embeddings, allow_dangerous_deserialization=True)
    llm = ChatOpenAI(model_name="gpt-3.5-turbo", temperature=0)

    query = ConversationalRetrievalChain.from_llm(
        llm=llm, 
        retriever=new_db.as_retriever(), 
        return_source_documents=True)
    return query({"question": question, "chat_history": chat_history})


def show_ui():

    st.title("Amazon Employee Benefits Chatbot")    
    st.subheader("Please enter your query about Amazon Employee Benefits")

    if "messages" not in st.session_state:
        st.session_state.messages = []
        st.session_state.chat_history = []

    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    if prompt := st.chat_input("Enter your Amazon Employee query: "):
        with st.spinner("Working on your query...."):     
            response = query(question=prompt, chat_history=st.session_state.chat_history)            
            with st.chat_message("user"):
                st.markdown(prompt)
            with st.chat_message("assistant"):
                st.markdown(response["answer"])    

            # Append user message to chat history
            st.session_state.messages.append({"role": "user", "content": prompt})
            st.session_state.messages.append({"role": "assistant", "content": response["answer"]})
            st.session_state.chat_history.extend([(prompt, response["answer"])])

if __name__ == "__main__":
    show_ui() 
