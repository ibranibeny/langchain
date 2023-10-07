from dotenv import load_dotenv
import streamlit as st
from PyPDF2 import PdfReader
from langchain.text_splitter import CharacterTextSplitter
import requests
from bs4 import BeautifulSoup
from langchain.document_loaders import TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import FAISS
from langchain.chains.question_answering import load_qa_chain
from langchain import HuggingFaceHub
from langchain.text_splitter import CharacterTextSplitter
from langchain.embeddings import OpenAIEmbeddings, HuggingFaceInstructEmbeddings, HuggingFaceEmbeddings
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationalRetrievalChain
from langchain.chat_models import ChatOpenAI
import os

def main():
    os.environ["HUGGINGFACEHUB_API_TOKEN"] = "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"
    load_dotenv()
    st.set_page_config(page_title="Ask your PDF")
    st.header("Ask your PDF 💬")
    
    # upload file
    pdf = st.file_uploader("Upload your PDF", type="pdf")
    
    # extract the text
    if pdf is not None:
      pdf_reader = PdfReader(pdf)
      text = ""
      for page in pdf_reader.pages:
        text += page.extract_text()
        
      #st.write(text)
      
      # split into chunks
      text_splitter = CharacterTextSplitter(
        separator="\n",
        chunk_size=1000,
        chunk_overlap=200,
        length_function=len
      )
      text_chunks = text_splitter.split_text(text)
      st.write(text_chunks)    

    # For Huggingface Embeddings

    embeddings = HuggingFaceInstructEmbeddings(model_name = "hkunlp/instructor-xl")
    #vector_store = FAISS.from_texts(texts = text_chunks, embedding = embeddings)
    vector_store = FAISS.from_texts(text_chunks, embeddings)
    #st.write(vectorstore)
    

    # HuggingFace Model

    llm = HuggingFaceHub(repo_id="tiiuae/falcon-40b-instruct", model_kwargs={"temperature":0.5, "max_length":512})
    memory = ConversationBufferMemory(memory_key='chat_history', return_messages=True)
    conversation_chain = ConversationalRetrievalChain.from_llm(
        llm = llm,
        retriever = vector_store.as_retriever(),
        memory = memory
    )
    st.write("DONE")

    # Create conversation chain
    st.session_state.conversation =  get_conversation_chain(vector_store)

if __name__ == '__main__':
    main()
