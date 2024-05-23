import os
from io import BytesIO
from dotenv import load_dotenv, find_dotenv
import flask
from flask import render_template, request
import ast  # for converting embeddings saved as strings back to arrays
import pandas as pd  # text and embeddings storage
import numpy #for the vector similarity calculation
from langchain_openai import OpenAI, ChatOpenAI
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import FAISS
from langchain.chat_models import azure_openai
from langchain.memory import ConversationBufferMemory
from langchain.chains import LLMChain
from langchain_core.prompts import (
    ChatPromptTemplate,
    MessagesPlaceholder,
    SystemMessagePromptTemplate,
    HumanMessagePromptTemplate,
)
# from langchain.embeddings import gpt4all
from langchain.chains.question_answering import load_qa_chain
import PyPDF2
from PyPDF2 import PdfReader
from langchain_community.document_loaders import PyPDFLoader
from pdfminer.high_level import extract_text
import tiktoken  # to count tokens
from scipy import spatial
import requests
import fitz
from langchain_community.vectorstores import AstraDB
from langchain.retrievers.self_query.astradb import AstraDBTranslator
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain_core.documents import Document

# import Chroma
# from Chroma import OpenAIEmbeddings

##################
# S E T T I N G S
##################

load_dotenv(find_dotenv())
the_key = os.getenv("BU_OPENAI_API_KEY")
astradb_key = os.getenv("ASTRADB_TOKEN")
astradb_api_endpoint = os.getenv("ASTRADB_API_ENDPOINT")
pdf_target_url = "https://api.akabot-staging.zetta-demo.space/fileuploads/Artificial-Intelligence-in-Finance-6a364d95-f26c-41e6-a3a1-54f9b9f975d2.pdf"


# models
EMBEDDING_MODEL = "text-embedding-ada-002"
GPT_MODEL = "gpt-3.5-turbo"
llm = OpenAI(api_key=os.getenv("OPEN_API_KEY"))
prompt = ChatOpenAI(
    messages=[
        MessagesPlaceholder(variable_name='chat_memory'),
        HumanMessagePromptTemplate.from_template("{question}")
    ]
)
embedding_model_instance = OpenAIEmbeddings(api_key=the_key, model=EMBEDDING_MODEL)


vstore = AstraDB(
    embedding=embedding_model_instance,
    collection_name="astra_vector_introduction",
    api_endpoint=astradb_api_endpoint,
    token=astradb_key
)

app = flask.Flask(__name__)


DATA_PATH = r"C:\Users\etocha\Documents\Zettabyte courses\RAG_Langchain_Training\data\NLP.pdf"
memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True, human_prefix="You >", ai_prefix="Bot >")


####################
# F U N C T I O N S
####################


def load_pdf():
    response = requests.get(pdf_target_url)
    if response.status_code == 200:
        myfile = BytesIO(response.content)
    doc = fitz.open(stream=myfile, filetype="pdf")
    print(f"Number of pages: {doc.page_count}")
    return doc

def extract_text_from_pdf(doc):
    texts = []
    for page_num in range(doc.page_count):
        page = doc.load_page(page_num)
        text = page.get_text("text").replace("\n", "")
        texts.append({
            "page_number": page_num + 1,  
            "page_content": text
        })
    return texts

def split_text(documents):
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1024,
        chunk_overlap=128,
        length_function=len,
        add_start_index=True
    )
    chunks_with_metadata = []
    for document in documents:
        chunks = text_splitter.split_text(document["page_content"])
        for i, chunk in enumerate(chunks):
            chunks_with_metadata.append({
                "chunk_id": f"{document['page_number']}_{i}",
                "content": chunk,
                "page_number": document['page_number'],
            })

    print(f"Split {len(documents)} documents into {len(chunks_with_metadata)} chunks.")
    return chunks_with_metadata

def custom_retriever(query, embedding_model_instance, vstore):
    
    results = vstore.similarity_search_with_score(query, k=1)
    for res, score in results:
        print(f"* [SIM={score:3f}] {res.page_content} [{res.metadata}]" )

    return f"* [SIM={score:3f}] {res.page_content} [{res.metadata}]"

######################################
# R E T R I E V I N G   P R O C E S S
######################################

def evaluate_similarity(query):
    results = vstore.similarity_search_with_score(query, k=1)
    for res, score in results:
        print(f"score : {score}")

    return score

def retrieve_that(input):
    doc = load_pdf()
    documents = extract_text_from_pdf(doc)
    splitted_text = split_text(documents)

    for i, chunk in enumerate(splitted_text[:5]):
        print(f"\nChunk {i}: {chunk}\n")

    documents_to_store = [Document(page_content=chunk["content"], metadata={"chunk_id": chunk["chunk_id"], "page_number": chunk["page_number"]}) for chunk in splitted_text]

    # vstore.add_documents(documents_to_store) # To use once

    query = input
    results = custom_retriever(query, embedding_model_instance, vstore)

    return results
    



################
# R U N N I N G
################


@app.route('/', methods= ['POST', 'GET'])
def home():
    return render_template('chat.html')

@app.route('/user_prompt_add', methods=['POST'])
def add_user_prompt():
    print("processing...")
    if request.method == 'POST':
        prompt_input = request.form["msg"]
        try : 
            score = evaluate_similarity(prompt_input)
            if score > 0.9:
                prompt_output = retrieve_that(prompt_input)
            else :
                prompt_output = llm.invoke(prompt_input)
        except Exception as e:
            prompt_output = f"Sorry, there was a problem connecting to the AI model. Please try later.\n {e}"
        print(prompt_input)
        print(prompt_output)
        memory.chat_memory.add_user_message(prompt_input)
        memory.chat_memory.add_ai_message(prompt_output)
        return render_template('chat.html', memory=memory.chat_memory, len = len(memory.chat_memory.messages))
    elif request.method == 'GET':
        return render_template('chat.html')
    
if __name__ == "__main__":
    app.run(debug=True)