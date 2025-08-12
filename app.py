import json
import os
import sys
import boto3
import streamlit as st
from botocore.exceptions import ClientError
from dotenv import load_dotenv

# FIX: Import GoogleDriveLoader and related dependencies
from langchain_community.document_loaders import GoogleDriveLoader, PyPDFDirectoryLoader
from langchain_community.embeddings import BedrockEmbeddings
from langchain_community.llms.bedrock import Bedrock
from langchain_pinecone import PineconeVectorStore
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.prompts import PromptTemplate
from langchain.chains import RetrievalQA

from pinecone import Pinecone as PineconeClient

# Load environment variables
load_dotenv()

# FIX: Revert to using st.secrets for Streamlit Cloud deployment
# The os.getenv approach is less reliable on Streamlit Cloud.
aws_access_key_id = st.secrets["AWS_ACCESS_KEY_ID"]
aws_secret_access_key = st.secrets["AWS_SECRET_ACCESS_KEY"]
aws_region_name = st.secrets.get("AWS_DEFAULT_REGION", "us-east-1") # Use a default region if not specified

# Set up the Bedrock client and embeddings model
bedrock = boto3.client(
    service_name="bedrock-runtime", 
    region_name=aws_region_name,
    aws_access_key_id=aws_access_key_id,
    aws_secret_access_key=aws_secret_access_key
)
bedrock_embeddings = BedrockEmbeddings(model_id="amazon.titan-embed-text-v2:0", client=bedrock)

# Setup Pinecone with your API key from environment variables
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
pc = PineconeClient(api_key=PINECONE_API_KEY)

INDEX_NAME = "langchain"  # Pinecone index name

# FIX: Update data_ingestion to use GoogleDriveLoader with credentials
def data_ingestion():
    # To use Google Drive Loader, you need to set up a service account and save the
    # key file as `credentials.json`.
    # Make sure to set the `GOOGLE_APPLICATION_CREDENTIALS` environment variable in Streamlit secrets
    # to point to this file.
    # IMPORTANT: Replace 'YOUR_FOLDER_ID' with the actual ID of your Google Drive folder.
    # The folder ID is a long string of letters and numbers, not a human-readable name.
    GOOGLE_DRIVE_FOLDER_ID = "RAG_APPLICATION"

    loader = GoogleDriveLoader(
        folder_id=GOOGLE_DRIVE_FOLDER_ID, 
        recursive=False
    )
    documents = loader.load()

    text_splitter = RecursiveCharacterTextSplitter(chunk_size=10000, chunk_overlap=1000)
    docs = text_splitter.split_documents(documents)
    return docs

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
You are an assistant that answers questions based only on the provided context.

<context>
{context}
</context>

Question: {question}

Instructions:
- Combine all the relevant information from the context related to the question.
- Do not guess or add information that is not in the context.
- If nothing in the context is relevant, reply exactly: "I don't have that information."
- Give a detailed but clear answer.

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

    user_question = st.text_input("Here to help with your queries...")

    with st.sidebar:
        st.title("Update Documents:")

        if st.button("Update"):
            with st.spinner("Processing..."):
                docs = data_ingestion()
                if docs:
                    get_vector_store(docs)
                    st.success("✅ You are up-to-date.")

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
