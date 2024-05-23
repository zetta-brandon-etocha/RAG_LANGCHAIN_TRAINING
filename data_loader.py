import requests
from io import BytesIO
import fitz
from langchain.text_splitter import RecursiveCharacterTextSplitter

pdf_target_url = "https://api.akabot-staging.zetta-demo.space/fileuploads/Artificial-Intelligence-in-Finance-6a364d95-f26c-41e6-a3a1-54f9b9f975d2.pdf"

def load_pdf():
    response = requests.get(pdf_target_url)
    if response.status_code == 200:
        myfile = BytesIO(response.content)
    doc = fitz.open(stream=myfile, filetype="pdf")
    return doc

def extract_text_from_pdf(doc):
    texts = []
    for page_num in range(doc.page_count):
        page = doc.load_page(page_num)
        text = page.get_text("text").replace("\n", "")
        texts.append({"page_number": page_num + 1, "page_content": text})
    return texts

def split_text(documents):
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=100,
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
    return chunks_with_metadata
