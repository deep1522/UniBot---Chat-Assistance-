import json
import os
import sys
import boto3
import streamlit as st

# CHANGE: Using the recommended langchain_pinecone library for vector store integration.
from langchain_pinecone import PineconeVectorStore
from langchain_community.embeddings import BedrockEmbeddings
from langchain_community.llms.bedrock import Bedrock

from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFDirectoryLoader

from langchain.prompts import PromptTemplate
from langchain.chains import RetrievalQA

# Import the Pinecone client to manage the index, but not for from_documents.
from pinecone import Pinecone as PineconeClient
from dotenv import load_dotenv
load_dotenv()

# Set up the Bedrock client and embeddings model
bedrock = boto3.client(service_name="bedrock-runtime")
# Note: The 'amazon.titan-embed-text-v2:0' model produces vectors with a dimension of 1024.
bedrock_embeddings = BedrockEmbeddings(model_id="amazon.titan-embed-text-v2:0", client=bedrock)

# Setup Pinecone with your API key from environment variables
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")

# Initialize the Pinecone client
pc = PineconeClient(api_key=PINECONE_API_KEY)

INDEX_NAME = "langchain"  # Pinecone index name

# Data ingestion remains unchanged
def data_ingestion():
    loader = PyPDFDirectoryLoader("data")
    documents = loader.load()

    text_splitter = RecursiveCharacterTextSplitter(chunk_size=10000, chunk_overlap=1000)
    docs = text_splitter.split_documents(documents)
    return docs

# CHANGE: This function now uses the correct PineconeVectorStore class
# from the langchain_pinecone library to ingest documents.
def get_vector_store(docs):
    # This static method handles the entire process: creating embeddings
    # and uploading them to the Pinecone index.
    # IMPORTANT: The Pinecone index dimension must match the embedding model's dimension (1024).
    vectorstore_pinecone = PineconeVectorStore.from_documents(
        docs,
        embedding=bedrock_embeddings,
        index_name=INDEX_NAME,
    )
    return vectorstore_pinecone

def get_claude_llm():
    llm = Bedrock(model_id="ai21.j2-mid-v1", client=bedrock, model_kwargs={'maxTokens': 512})
    return llm

def get_llama2_llm():
    # Corrected model ID for llama3-70b
    llm = Bedrock(model_id="meta.llama3-70b-instruct-v1:0", client=bedrock, model_kwargs={'max_gen_len': 512})
    return llm

# prompt_template = """
# Human: Use the following pieces of context to provide a 
# concise answer to the question at the end but use at least summarize with 
# 250 words with detailed explanations. If you don't know the answer, 
# just say that you don't know, don't try to make up an answer.
# <context>
# {context}
# </context>

# Question: {question}

# Assistant:"""

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
                # upload embeddings to Pinecone index
                get_vector_store(docs)
                st.success("✅ You are up-to-date.")

    if st.button("Search"):
        with st.spinner("Processing..."):
            # CHANGE: Load the vector store from an existing Pinecone index.
            # This is the correct method for the new PineconeVectorStore class.
            vectorstore_pinecone = PineconeVectorStore.from_existing_index(
                index_name=INDEX_NAME,
                embedding=bedrock_embeddings
            )
            llm = get_llama2_llm()
            st.write(get_response_llm(llm, vectorstore_pinecone, user_question))
            st.success("✅ Anything Else?")

if __name__ == "__main__":
    main()