import json
import os
import sys
import boto3
import streamlit as st
from botocore.exceptions import ClientError
from dotenv import load_dotenv

# FIX: Import the service_account module for a better way to handle credentials
from google.oauth2 import service_account
# FIX: Use the new, non-deprecated GoogleDriveLoader
from langchain_google_community import GoogleDriveLoader
from langchain_community.embeddings import BedrockEmbeddings
from langchain_community.llms.bedrock import Bedrock
from langchain_pinecone import PineconeVectorStore
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.prompts import PromptTemplate
from langchain.chains import RetrievalQA

from pinecone import Pinecone as PineconeClient

# Load environment variables
load_dotenv()

# --- AWS & Pinecone Setup ---
# A function to get credentials from environment variables (for local development)
def get_aws_credentials():
    aws_access_key_id = os.getenv("AWS_ACCESS_KEY_ID")
    aws_secret_access_key = os.getenv("AWS_SECRET_ACCESS_KEY")
    return aws_access_key_id, aws_secret_access_key

def get_pinecone_api_key():
    pinecone_api_key = os.getenv("PINECONE_API_KEY")
    return pinecone_api_key

aws_access_key_id, aws_secret_access_key = get_aws_credentials()
aws_region_name = os.getenv("AWS_DEFAULT_REGION", "us-east-1") # Use a default region if not specified

# Set up the Bedrock client and embeddings model
bedrock = boto3.client(
    service_name="bedrock-runtime", 
    region_name=aws_region_name,
    aws_access_key_id=aws_access_key_id,
    aws_secret_access_key=aws_secret_access_key
)
bedrock_embeddings = BedrockEmbeddings(model_id="amazon.titan-embed-text-v2:0", client=bedrock)

# Setup Pinecone with your API key
PINECONE_API_KEY = get_pinecone_api_key()
pc = PineconeClient(api_key=PINECONE_API_KEY)

INDEX_NAME = "langchain"  # Pinecone index name

# --- Google Drive Credentials Setup ---
def setup_google_credentials():
    """
    Handles Google credentials for local environments by checking for a credentials.json file.
    """
    credentials_file = os.getenv("GOOGLE_APPLICATION_CREDENTIALS")
    if credentials_file and os.path.exists(credentials_file):
        try:
            credentials = service_account.Credentials.from_service_account_file(credentials_file)
            st.success("Google credentials set up from local file.")
            return credentials
        except Exception as e:
            st.error(f"Error loading local credentials.json file. Details: {e}")
            return None
    else:
        st.error("`credentials.json` file not found or GOOGLE_APPLICATION_CREDENTIALS not set for local development.")
        return None

# --- Application Functions ---
def data_ingestion(gdrive_credentials):
    if gdrive_credentials is None:
        st.error("Google Drive credentials are not available. Document ingestion failed.")
        return []

    GOOGLE_DRIVE_FOLDER_ID = "1ZB8Ur70bjRoZxrNOSOaxS6cSjcv6V1nN"
    
    loader = GoogleDriveLoader(
        folder_id=GOOGLE_DRIVE_FOLDER_ID,
        credentials=gdrive_credentials,
        recursive=False
    )
    
    try:
        documents = loader.load()
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=10000, chunk_overlap=1000)
        docs = text_splitter.split_documents(documents)
        return docs
    except Exception as e:
        st.error(f"Error loading documents from Google Drive. Details: {e}")
        return []

def get_vector_store(docs):
    try:
        vectorstore_pinecone = PineconeVectorStore.from_documents(
            docs,
            embedding=bedrock_embeddings,
            index_name=INDEX_NAME,
        )
        return vectorstore_pinecone
    except Exception as e:
        st.error(
            f"An error occurred while creating the vector store. "
            f"Please check your AWS and Pinecone configurations. "
            f"Full error: {e}"
        )
        return None

def get_claude_llm():
    llm = Bedrock(model_id="ai21.j2-mid-v1", client=bedrock, model_kwargs={'maxTokens': 512})
    return llm

def get_llama2_llm():
    llm = Bedrock(model_id="meta.llama3-70b-instruct-v1:0", client=bedrock, model_kwargs={'max_gen_len': 512})
    return llm

prompt_template = """
You are an assistant that answers questions based ONLY on the provided context.
Answer with information that is present in the context.

<context>
{context}
</context>

Question: {question}

Instructions:
- If the question cannot be answered from the given context, you MUST reply with "I do not have that information.".
- Do NOT use any external knowledge.
- Your answer must be a single, detailed summary based on the relevant parts of the context.
- Your answer must be truthful and accurate to the provided context.

Assistant:
"""

PROMPT = PromptTemplate(template=prompt_template, input_variables=["context", "question"])

def get_response_llm(llm, vectorstore, query):
    qa = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=vectorstore.as_retriever(search_type="similarity", search_kwargs={"k": 3}),
        return_source_documents=True,
        chain_type_kwargs={"prompt": PROMPT}
    )
    answer = qa({"query": query})
    return answer['result']

def main():
    st.set_page_config("Chat PDF with Pinecone")

    st.header("UNIBOT - AWS Bedrock + Pinecone")

    gdrive_credentials = setup_google_credentials()
    
    # Initialize session state for messages
    if 'status_message' not in st.session_state:
        st.session_state['status_message'] = ""
    if 'status_type' not in st.session_state:
        st.session_state['status_type'] = "info"

    user_question = st.text_input("Here to help with your queries...")

    with st.sidebar:
        st.title("Update Documents:")

        if st.button("Update"):
            st.session_state['status_message'] = ""
            st.session_state['status_type'] = "info"
            with st.spinner("Processing..."):
                docs = data_ingestion(gdrive_credentials)
                if docs:
                    get_vector_store(docs)
                    st.session_state['status_message'] = "✅ You are up-to-date."
                    st.session_state['status_type'] = "success"
                else:
                    st.session_state['status_message'] = "❌ Document ingestion failed. Please check the logs."
                    st.session_state['status_type'] = "error"
        
        # FIX: Move the status message display to the sidebar
        if st.session_state['status_message']:
            if st.session_state['status_type'] == "success":
                st.success(st.session_state['status_message'])
            else:
                st.error(st.session_state['status_message'])

    if st.button("Search"):
        with st.spinner("Processing..."):
            vectorstore_pinecone = PineconeVectorStore.from_existing_index(
                index_name=INDEX_NAME,
                embedding=bedrock_embeddings
            )
            llm = get_llama2_llm()
            st.write(get_response_llm(llm, vectorstore_pinecone, user_question))
            st.success("✅ Anything Else?")

if __name__ == "__main__":
    main()
