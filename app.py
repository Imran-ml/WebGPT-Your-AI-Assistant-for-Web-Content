import streamlit as st
from langchain_core.messages import AIMessage, HumanMessage
from langchain_community.document_loaders.web_base import WebBaseLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores.chroma import Chroma
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from dotenv import load_dotenv
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain.chains.history_aware_retriever import create_history_aware_retriever
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.chains.retrieval import create_retrieval_chain

load_dotenv()

def get_vectorstore_from_url(url):
    loader = WebBaseLoader(url)
    document = loader.load()
    text_splitter = RecursiveCharacterTextSplitter()
    document_chunks = text_splitter.split_documents(document)

    # Create a vector store of embeddings
    vector_store = Chroma.from_documents(document_chunks, OpenAIEmbeddings())
    return vector_store

def get_contect_retriever_chain(vector_store):
    llm = ChatOpenAI()
    retriever = vector_store.as_retriever()

    # Define a prompt template for the retriever chain
    prompt = ChatPromptTemplate.from_messages([
        MessagesPlaceholder(variable_name="chat_history"),
        ("user", "{input}"),
        ("user", "Given the above conversation, generate a search query to look up in order to get information relevant to the conversation.")
    ])

    retriever_chain = create_history_aware_retriever(llm, retriever, prompt)
    return retriever_chain

def get_conversational_rag_chain(retrieval_chain):
    llm = ChatOpenAI()

    # Define a prompt template for the RAG chain
    prompt = ChatPromptTemplate.from_messages([
        ("system", "Answer the user's question based on the below context :\n\n {context}"),
        MessagesPlaceholder(variable_name="chat_history"),
        ("user", "{input}"),
    ])
    stuff_documents_chain = create_stuff_documents_chain(llm, prompt)

    return create_retrieval_chain(retrieval_chain, stuff_documents_chain)

def get_response(user_input):
    retriever_chain = get_contect_retriever_chain(st.session_state.vector_store)
    conversation_rag_chain = get_conversational_rag_chain(retriever_chain)
    
    response = conversation_rag_chain.invoke({
        "chat_history": st.session_state.chat_history,
        "input": user_input
    })
    
    return response['answer']

# Streamlit app configuration
st.set_page_config(page_title = "WebGPT", page_icon = "🌐", layout = "wide")
st.title("WebGPT: Your AI Assistant for Web Content")

with st.sidebar:
    # st.header("Website URL")
    website_url = st.text_input("Website URL", placeholder="Enter a website URL")

if website_url is None or website_url == "":
    st.info("Please enter a website URL to start the conversation...")
else:
    # Initialize session state variables
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = [
            AIMessage(content="Hi, I'm WebGPT. How can I help you?"),
        ]
        
    if "vector_store" not in st.session_state:
        st.session_state.vector_store = get_vectorstore_from_url(website_url)
    
    # User input
    user_query = st.chat_input("Message WebGPT...")
    if user_query is not None and user_query != "":
        response = get_response(user_query)
        st.session_state.chat_history.append(HumanMessage(content=user_query))
        st.session_state.chat_history.append(AIMessage(content=response))
        
    # Display conversation history
    for message in st.session_state.chat_history:
        if isinstance(message, AIMessage):
            with st.chat_message("AI"):
                st.write(message.content)
        if isinstance(message, HumanMessage):
            with st.chat_message("Human"):
                st.write(message.content)
