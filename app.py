import flask, os
from flask import render_template, request
from langchain_openai import OpenAI
from langchain.chains.question_answering import load_qa_chain
from langchain.memory import ConversationBufferMemory
from data_loader import load_pdf, extract_text_from_pdf, split_text
from db_manager import store_documents, custom_retriever, evaluate_similarity

app = flask.Flask(__name__)
llm = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True, human_prefix="You >", ai_prefix="Bot >")

####################
# F U N C T I O N S
####################

def db_manager_process():
    doc = load_pdf()
    documents = extract_text_from_pdf(doc)
    splitted_text = split_text(documents)
    store_documents(splitted_text)


def retrieve_that(input):
    results = custom_retriever(input, k=5)
    documents = [res for res, score in results]
    qa_chain = load_qa_chain(llm, chain_type="stuff")
    response = qa_chain.run(input_documents=documents, question=input)
    return response

##############
# R O U T E S
##############

@app.route('/', methods=['POST', 'GET'])
def home():
    return render_template('chat.html')

@app.route('/user_prompt_add', methods=['POST'])
def add_user_prompt():
    #######################
    # only one time
    # db_manager_process()
    #######################
    if request.method == 'POST':
        prompt_input = request.form["msg"]
        try:
            score = evaluate_similarity(prompt_input)
            if score > 0.9:
                prompt_output = retrieve_that(prompt_input) + " (from PDF)"
            else:
                prompt_output = llm.invoke(prompt_input) + " (from OpenAI)"
        except Exception as e:
            prompt_output = f"Sorry, there was a problem connecting to the AI model. Please try later.\n {e}"
        memory.chat_memory.add_user_message(prompt_input)
        memory.chat_memory.add_ai_message(prompt_output)
        return render_template('chat.html', memory=memory.chat_memory, len=len(memory.chat_memory.messages))
    elif request.method == 'GET':
        return render_template('chat.html')

if __name__ == "__main__":
    app.run(debug=True)
