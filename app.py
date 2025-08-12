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

# Load environment variables (for local dev)
load_dotenv()

# --- AWS & Pinecone Setup ---
def get_aws_credentials():
    # Use st.secrets on Streamlit Cloud, fallback to os.getenv locally
    aws_access_key_id = st.secrets.get("AWS_ACCESS_KEY_ID") or os.getenv("AWS_ACCESS_KEY_ID")
    aws_secret_access_key = st.secrets.get("AWS_SECRET_ACCESS_KEY") or os.getenv("AWS_SECRET_ACCESS_KEY")
    return aws_access_key_id, aws_secret_access_key

def get_pinecone_api_key():
    return st.secrets.get("PINECONE_API_KEY") or os.getenv("PINECONE_API_KEY")

aws_access_key_id, aws_secret_access_key = get_aws_credentials()
aws_region_name = st.secrets.get("AWS_DEFAULT_REGION", "us-east-1") or os.getenv("AWS_DEFAULT_REGION", "us-east-1")

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
    Handles Google credentials for both cloud and local environments.
    """
    credentials_source = st.secrets.get("GOOGLE_APPLICATION_CREDENTIALS") or os.getenv("GOOGLE_APPLICATION_CREDENTIALS")
    
    if credentials_source:
        # FIX: Corrected logic to try JSON string first, then fall back to file path
        try:
            # First, assume it's a JSON string (Streamlit Cloud)
            credentials_dict = json.loads(credentials_source)
            credentials = service_account.Credentials.from_service_account_info(credentials_dict)
            st.success("Google credentials set up from Streamlit secrets.")
            return credentials
        except json.JSONDecodeError:
            # If it's not a JSON string, assume it's a file path (local development)
            if os.path.exists(credentials_source):
                try:
                    credentials = service_account.Credentials.from_service_account_file(credentials_source)
                    st.success("Google credentials set up from local file.")
                    return credentials
                except Exception as e:
                    st.error(f"Error loading local credentials file. Details: {e}")
                    return None
            else:
                st.error(f"Error: Credentials source is not a valid JSON string and file at path '{credentials_source}' was not found.")
                return None
        except Exception as e:
            st.error(f"An unexpected error occurred with Google credentials. Details: {e}")
            return None
    else:
        st.error("`GOOGLE_APPLICATION_CREDENTIALS` not found in secrets or environment.")
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
        # FIX: Changed chunk size and overlap for better retrieval performance
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=100)
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
You are an assistant that provides comprehensive and truthful answers based ONLY on the provided context.
Your goal is to use all relevant information from the documents to answer the user's query as thoroughly as possible.

<context>
{context}
</context>

Question: {question}

Instructions:
- If the question can be answered from the provided context, include ALL relevant details.
- Do NOT use any information from outside of the provided context.
- If the context does not contain any information relevant to the question, you MUST reply with "I do not have that information.".
- Your answer must be factual and directly supported by the context.

Assistant:
"""

PROMPT = PromptTemplate(template=prompt_template, input_variables=["context", "question"])

def get_response_llm(llm, vectorstore, query):
    # FIX: Increased k to retrieve more documents for a more comprehensive answer
    qa = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=vectorstore.as_retriever(search_type="similarity", search_kwargs={"k": 10}),
        return_source_documents=True,
        chain_type_kwargs={"prompt": PROMPT}
    )
    answer = qa({"query": query})
    return answer['result']

def main():
    st.set_page_config("Chat PDF with Pinecone")

    st.header("UNIBOT - AWS Bedrock + Pinecone")
    
    # Initialize session state for messages
    if 'status_message' not in st.session_state:
        st.session_state['status_message'] = ""
    if 'status_type' not in st.session_state:
        st.session_state['status_type'] = "info"
    if 'user_question_input' not in st.session_state:
        st.session_state.user_question_input = ""

    gdrive_credentials = setup_google_credentials()

    user_question = st.text_input("Here to help with your queries...", key="user_question_input")

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
            response = get_response_llm(llm, vectorstore_pinecone, user_question)
            
            # Display the question and response
            st.write(f"**Question:** {user_question}")
            st.write(response)
            
            # FIX: Clear the input field for next queries
            st.session_state.user_question_input = ""
            
            st.success("✅ Anything Else?")

if __name__ == "__main__":
    main()
