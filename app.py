import gradio as gr
import fitz
from docx import Document
import numpy as np
import faiss
from sentence_transformers import SentenceTransformer
import requests
from google.colab import userdata

# Load API Key
groq_api_key = userdata.get('RAG-API')

# Load embedding model
embedder = SentenceTransformer("all-MiniLM-L6-v2")

# Helper functions
def extract_text(file_path):
    if file_path.endswith(".pdf"):
        with fitz.open(file_path) as doc:
            return "\n".join(page.get_text() for page in doc)
    elif file_path.endswith(".docx"):
        doc = Document(file_path)
        return "\n".join(p.text for p in doc.paragraphs)
    else:
        raise ValueError("Unsupported file format")

def chunk_text(text, chunk_size=500):
    return [text[i:i+chunk_size] for i in range(0, len(text), chunk_size)]

def get_embeddings(chunks):
    return embedder.encode(chunks)

def create_faiss_index(embeddings):
    dim = embeddings.shape[1]
    index = faiss.IndexFlatL2(dim)
    index.add(embeddings)
    return index

def search_faiss(query, chunks, embedder, index, top_k=3):
    query_vec = embedder.encode([query])
    D, I = index.search(query_vec, top_k)
    return [chunks[i] for i in I[0]]

def query_llama3(prompt):
    url = "https://api.groq.com/openai/v1/chat/completions"
    headers = {"Authorization": f"Bearer {groq_api_key}"}
    data = {
        "model": "llama3-70b-8192",
        "messages": [{"role": "user", "content": prompt}],
        "temperature": 0.7
    }
    response = requests.post(url, headers=headers, json=data)
    return response.json()['choices'][0]['message']['content']

def rag_pipeline(file, question):
    try:
        file_path = file.name
        text = extract_text(file_path)
        chunks = chunk_text(text)
        embeddings = get_embeddings(chunks)
        faiss_index = create_faiss_index(np.array(embeddings))
        relevant = search_faiss(question, chunks, embedder, faiss_index)
        context = "\n".join(relevant)

        prompt = f"Answer the question using the following context:\n\n{context}\n\nQuestion: {question}"
        answer = query_llama3(prompt)
        return answer

    except Exception as e:
        return f"❌ Error: {str(e)}"

# Gradio Interface
with gr.Blocks(title="🧠 Visual Intelligence RAG System") as demo:
    gr.Markdown("""
    <h1 style='text-align: center; font-size: 28px; margin-bottom: 20px;'>
        🧠 Visual Intelligence RAG System (GROQ LLaMA3)
    </h1>
    """)

    with gr.Row():
        with gr.Column(scale=1):
            upload = gr.File(label="📄 Upload PDF or Word", file_types=[".pdf", ".docx"], height=150)
            gr.Markdown("""
            <div style="font-size: 14px; line-height: 1.6; margin-top: 10px;">
                👨‍💻 <strong>Developer:</strong> M. Shahjhan Gondal<br>
                📧 <strong>Email:</strong> <a href="mailto:shahjhangondal99@gmail.com">shahjhangondal99@gmail.com</a><br>
                🌐 <strong>GitHub:</strong> <a href="https://github.com/shahjhan99" target="_blank">https://github.com/shahjhan99</a>
            </div>
            """)

        with gr.Column(scale=1):
            query = gr.Textbox(label="Your Query❓", lines=2)
            run_button = gr.Button("📌 Run", size="sm")
            output = gr.Textbox(label="📘 Answer", lines=10)
            flag_button = gr.Button("🚩 Flag", size="sm")

    # Function binding
    run_button.click(fn=rag_pipeline, inputs=[upload, query], outputs=output)

# Launch
demo.launch()
