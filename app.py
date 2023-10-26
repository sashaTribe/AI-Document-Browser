import streamlit as st
from dotenv import load_dotenv
from PyPDF2 import PdfReader
from langchain.text_splitter import RecursiveCharacterTextSplitter, CharacterTextSplitter
from langchain.embeddings import OpenAIEmbeddings, HuggingFaceInstructEmbeddings
from langchain.vectorstores.faiss import FAISS
from langchain.chat_models import ChatOpenAI
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationalRetrievalChain
from langchain.schema import (
    SystemMessage,
    HumanMessage,
    AIMessage
)
from streamlit_chat import message
import os
 
def get_pdf_text(pdf_docs):
    text = ""
    for pdf in pdf_docs:
        pdf_reader = PdfReader(pdf)
        for page in pdf_reader.pages:
            text +=page.extract_text()
    return text

def get_text_chunks(text):
    text_splitter = RecursiveCharacterTextSplitter(
        separators=["\n","\n\n"," ",""],
        chunk_size=1000,
        chunk_overlap=200,
        length_function=len
    )
    chunks = text_splitter.split_text(text)
    return chunks

def get_vectorstore(text_chunks):
    embeddings = OpenAIEmbeddings()
    vectorstore = FAISS.from_texts(texts=text_chunks, embedding=embeddings)
    return vectorstore

def get_conversation_chain(vectorstore):
    llm = ChatOpenAI()
    memory = ConversationBufferMemory(memory_key='chat_history', return_messages=True)
    conversation_chain = ConversationalRetrievalChain.from_llm(
        llm=llm,
        retriever=vectorstore.as_retriever(),
        memory=memory
    )
    return conversation_chain

def init():
    load_dotenv()
    # test that the API key exists
    if os.getenv("OPENAI_API_KEY") is None or os.getenv("OPENAI_API_KEY") == "":
        print("OPENAI_API_KEY is not set")
        exit(1)
    else:
        print("OPENAI_API_KEY is set")

    st.set_page_config(page_title="AI Document Reader Chat", page_icon=":books:")

def main():
    init()

    chat = ChatOpenAI(temperature=1)

    if "conversation" not in st.session_state:
        st.session_state.conversation = [
            SystemMessage(content="Hello there! How may I be of assistance?")
            ]

    st.header("AI Document Reader Chat")

    user_question = st.text_input("Ask a question about your document: ", key="user_question")
    if user_question and user_question not in [message.content for message in st.session_state.conversation]:
       st.session_state.conversation.append(HumanMessage(content=user_question))
       with st.spinner("Thinking..."):
            response = chat(st.session_state.conversation)
       st.session_state.conversation.append(AIMessage(content=response.content))
    
    conversation = st.session_state.get('conversation', [])
    for i, msg in enumerate(conversation[1:]):
        if i % 2 == 0:
            message(msg.content, is_user=True, key=str(i) + '_user')
        else:
            message(msg.content, is_user=False, key=str(i) + '_ai')

    with st.sidebar:
        st.subheader("Your Documents")
        pdf_docs = st.file_uploader(
            "Upload your data and click on Process", accept_multiple_files=True)
        if st.button("Process"):
            with st.spinner("Processing"):
                raw_text = get_pdf_text(pdf_docs)                                           # Turns PDF Into Text
                text_chunks = get_text_chunks(raw_text)                                     # Splits Text in chunks
                vectorstore = get_vectorstore(text_chunks)                                  # Store Vectors
                st.session_state.conversation = get_conversation_chain(vectorstore)    # Conversation Chain

if __name__ == '__main__':
    main()