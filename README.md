## 🚀 Project Overview

This project implements a Retrieval‑Augmented Generation (RAG) system that lets users upload PDF or Word documents and ask natural‑language questions against their content using FAISS vector search, Hugging Face embeddings, and LLaMA3 via the GROQ API in a Gradio interface. ([GitHub][3]) ([GitHub][4])

## 📦 Features

* **Document Upload**: Support for `.pdf` and `.docx` uploads. ([Gist][5])
* **Automatic Chunking & Embedding**: Text is split into 500‑character chunks and encoded with `all‑MiniLM‑L6‑v2`. ([FreeCodeCamp][1])
* **Vector Search**: FAISS performs efficient nearest‑neighbor retrieval over embeddings. ([GitHub Docs][2])
* **LLM Response**: GROQ’s LLaMA3‑70B generates answers conditioned on retrieved context. ([GitHub][3])
* **Secure Secrets**: API keys are managed via environment variables (no hard‑coding). ([Hatica][6])

## ⚙️ Installation

1. Clone the repository:

   ```bash
   git clone https://github.com/shahjhan99/visual-intelligence-rag-system.git
   ```
2. Install dependencies:

   ```bash
   pip install -r requirements.txt
   ```
3. (Optional) Create a virtual environment for isolation. ([Gist][5])

## 📋 Usage

1. Set your secret key as an environment variable:

   ```bash
   export RAG_SYSTEM_API="your_groq_api_key"
   ```
2. Launch the app:

   ```bash
   python app.py
   ```
3. Open the displayed Gradio URL in your browser, upload your document, enter a query, and click **Run**. ([GitHub Docs][2])

## 🔒 Secrets & Configuration

* **Environment Variable**: `RAG_SYSTEM_API` must be defined before running. ([readme-templates.com][7])
* **Hugging Face Spaces**: Add via **Settings → Secrets and variables** with key `RAG_SYSTEM_API`. ([GitHub Docs][2])

## 🤝 Contributing

We welcome contributions!

1. Fork the repo.
2. Create a feature branch (`git checkout -b feature/my-feature`).
3. Commit your changes (`git commit -m 'Add feature'`).
4. Push to your branch (`git push origin feature/my-feature`).
5. Open a Pull Request. ([GitHub][8])

## 📄 License

This project is released under the **MIT License**. See [LICENSE](LICENSE) for details. ([GitHub Docs][9])

## 📞 Contact

👤 **Developer**: M. Shahjhan Gondal
📧 **Email**: [shahjhangondal99@gmail.com](mailto:shahjhangondal99@gmail.com)
🌐 **GitHub**: [shahjhan99](https://github.com/shahjhan99)
