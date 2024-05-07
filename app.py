import os
from io import BytesIO
from dotenv import load_dotenv
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
# import Chroma
# from Chroma import OpenAIEmbeddings

load_dotenv()

# models
EMBEDDING_MODEL = "text-embedding-ada-002"
GPT_MODEL = "gpt-3.5-turbo"
llm = OpenAI(api_key=os.getenv("MY_OPENAI_API_KEY"))
prompt = ChatOpenAI(
    messages=[
        MessagesPlaceholder(variable_name='chat_memory'),
        HumanMessagePromptTemplate.from_template("{question}")
    ]
)
app = flask.Flask(__name__)


DATA_PATH = r"C:\Users\etocha\Documents\Zettabyte courses\RAG_Langchain_Training\data\NLP.pdf"
memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True, human_prefix="You >", ai_prefix="Bot >")

CHROMA_PATH = "chroma"

# db = Chroma.from_documents(chunks, OpenAIEmbeddings(), persist_directory=CHROMA_PATH)


def load_documents():
    loader = PyPDFLoader(DATA_PATH)
    documents = loader.load()

    return documents

def split_text(documents):
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size = 1000,
        chunk_overlap = 500,
        length_function = len, 
        add_start_index = True,
    )

    chunks = text_splitter.split_documents(documents)    
    print(f"Split {len(documents)} documents into {len(chunks)} chunks.")


    return chunks

def process_pdf_bis():
    text = extract_text("NLP.pdf")

    return text

@app.route('/', methods= ['POST', 'GET'])
def home():
    split_text(load_documents())
    return render_template('index.html')

@app.route('/user_prompt_add', methods=['POST'])
def add_user_prompt():
    if request.method == 'POST':
        prompt_input = request.form["user_input"]
        try : 
            prompt_output = llm.invoke(prompt_input)
        except Exception as e:
            prompt_output = f"Sorry, there was a problem connecting to the AI model. Please try later.\n {e}"
        print(prompt_output)
        memory.chat_memory.add_user_message(prompt_input)
        memory.chat_memory.add_ai_message(prompt_output)
        return render_template('index.html', memory=memory.chat_memory)
    elif request.method == 'GET':
        return render_template('index.html')
    
if __name__ == "__main__":
    app.run(debug=True)