from langchain.embeddings.openai import OpenAIEmbeddings
from langchain_core.documents import Document
from langchain_community.vectorstores import AstraDB
from dotenv import load_dotenv, find_dotenv
import os

load_dotenv(find_dotenv())
the_key = os.getenv("OPENAI_API_KEY")
astradb_key = os.getenv("ASTRADB_TOKEN")
astradb_api_endpoint = os.getenv("ASTRADB_API_ENDPOINT")

embedding_model_instance = OpenAIEmbeddings(api_key=the_key, model="text-embedding-ada-002")

vstore = AstraDB(
    embedding=embedding_model_instance,
    collection_name="astra_vector_introduction",
    api_endpoint=astradb_api_endpoint,
    token=astradb_key
)

def store_documents(documents):
    ###########################
    # resets the db
    # vstore.clear() 
    ###########################
    documents_to_store = [
        Document(page_content=chunk["content"], metadata={"chunk_id": chunk["chunk_id"], "page_number": chunk["page_number"]}) 
        for chunk in documents
    ]
    vstore.add_documents(documents_to_store)

def custom_retriever(query, k=5):
    results = vstore.similarity_search_with_score(query, k=k)
    return results

def evaluate_similarity(query):
    results = vstore.similarity_search_with_score(query, k=1)
    score = results[0][1]
    return score
