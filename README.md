# Creating an Document Summarizer (Agentic AI) with the help of DeepSeek R1 model provided by Ollama

# Project Goal : 
1. The idea is to build an autonomous document summarizer (a kind of agentic AI) that leverages the advanced reasoning capabilities of the DeepSeek R1 model, which is available via Ollama. 

2. Essentially, you extract text from your document, feed it into DeepSeek R1 running locally, and let the model “think” through the content using its chain-of-thought reasoning to produce a concise, high-quality summary—all with minimal human intervention.

# Workflow :
![image](https://github.com/user-attachments/assets/9df56184-cb97-4c6a-bdb4-4cdd01c34ae8)

# Working :

All the required libraries for project

    streamlit
    langchain_core
    langchain_community
    langchain_ollama
    pdfplumber

Then write the code in rag_deep.py file

    import streamlit as st
    from langchain_community.document_loaders import PDFPlumberLoader
    from langchain_text_splitters import RecursiveCharacterTextSplitter
    from langchain_core.vectorstores import InMemoryVectorStore
    from langchain_ollama import OllamaEmbeddings
    from langchain_core.prompts import ChatPromptTemplate
    from langchain_ollama.llms import OllamaLLM
    
    PROMPT_TEMPLATE = """ Your prompt """
    PDF_STORAGE_PATH = 'document_store/pdfs/'
    EMBEDDING_MODEL = OllamaEmbeddings(model="deepseek-r1:1.5b")
    DOCUMENT_VECTOR_DB = InMemoryVectorStore(EMBEDDING_MODEL)
    LANGUAGE_MODEL = OllamaLLM(model="deepseek-r1:1.5b")

    import os
    def save_uploaded_file(uploaded_file):
        # Define the directory where you want to save the file

        # Create the directory if it doesn't exist

        # Construct the full file path

        # file_path = PDF_STORAGE_PATH + uploaded_file.name

        return file_path

    def load_pdf_documents(file_path):
        document_loader = PDFPlumberLoader(file_path)
        return document_loader.load()

    def chunk_documents(raw_documents):
        text_processor = RecursiveCharacterTextSplitter ( )
        return text_processor

    def index_documents(document_chunks):
        DOCUMENT_VECTOR_DB.add_documents(document_chunks)

    def find_related_documents(query):
        return DOCUMENT_VECTOR_DB.similarity_search(query)

    def generate_answer(user_query, context_documents):   
        return response_chain.invoke({"user_query": user_query, "document_context": context_text})

    # UI Configuration
    st.title()
    st.markdown()

    # File Upload Section
    uploaded_pdf = st.file_uploader()

    if uploaded_pdf:
        saved_path
        raw_docs
        processed_chunks

        st.success("✅ Document processed successfully! Ask your questions below.")
        user_input = st.chat_input("Enter your question about the document...")

        if user_input:
            with st.chat_message("user"):
               st.write(user_input)
            with st.spinner("Analyzing document..."):
                relevant_docs = find_related_documents(user_input)
                ai_response = generate_answer(user_input, relevant_docs)
            with st.chat_message("assistant", avatar="🤖"):
               st.write(ai_response)

Now you have created your model code and streamlit ui

# Streamlit application
![Screenshot 2025-02-22 125030](https://github.com/user-attachments/assets/aef12e69-390b-4769-8423-b2dbc8e15367)

![Screenshot 2025-02-22 125055](https://github.com/user-attachments/assets/6da051d3-9683-4c36-9e0c-1e025dd8192e)


