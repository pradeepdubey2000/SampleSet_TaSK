# 🌟 Retrieval-Augmented Generation (RAG) Model for QA Bot 💡

## PART 1: Overview

🎯 **Objective:** Develop a RAG model for a QA bot using Pinecone DB and Cohere API for information retrieval and answer generation.

---

## 🚀 Requirements:
- ✅ Implement a RAG model for document-based Q&A.
- ✅ Use Pinecone for storing and retrieving embeddings.
- ✅ Test with various queries.

## 📦 Deliverables:
- 📑 Colab notebook showcasing the pipeline.
- 📘 Documentation on model architecture and response generation.
- 📊 Example queries and outputs.

---

## ⚙️ Part 1: Environment Setup

### 1️⃣ Install Libraries 📥
```bash
!pip install pinecone-client cohere transformers
```

### 2️⃣ Import Libraries 📚
```python
from pinecone import Pinecone
import cohere
from transformers import AutoTokenizer, AutoModel
import torch

# Initialize
pc = Pinecone(api_key="YOUR_PINECONE_API_KEY")
index = pc.Index("index384")
co = cohere.Client("YOUR_COHERE_API_KEY")
```

---

## ✏️ Part 2: Tokenization and Embeddings

### 3️⃣ Load Tokenizer and Model:
```python
tokenizer = AutoTokenizer.from_pretrained('sentence-transformers/all-MiniLM-L6-v2')
model = AutoModel.from_pretrained('sentence-transformers/all-MiniLM-L6-v2')

def embed_text(text):
    inputs = tokenizer(text, return_tensors='pt', padding=True, truncation=True)
    outputs = model(**inputs)
    embeddings = outputs.last_hidden_state.mean(dim=1)
    return embeddings.detach().numpy()[0]
```

---

## 📄 Part 3: Document Storage

### 4️⃣ Embed and Store Documents:
```python
documents = ["Document 1...", "Document 2..."]

for i, doc in enumerate(documents):
    vector = embed_text(doc)
    index.upsert(vectors=[(str(i), vector)])
```

---

## 🔍 Part 4: Retrieval

### 5️⃣ Retrieve Relevant Documents:
```python
def retrieve_documents(query, top_k=3):
    query_embedding = embed_text(query)
    results = index.query(vector=query_embedding.tolist(), top_k=top_k)
    return [documents[int(match['id'])] for match in results['matches']]
```

---

## 🤖 Part 5: Answer Generation

### 6️⃣ Generate Responses:
```python
def generate_answer(relevant_docs, query):
    context = " ".join(relevant_docs)
    response = co.generate(
        model='command-xlarge-nightly',
        prompt=f"Answer based on: {context}\nQuestion: {query}",
        max_tokens=100
    )
    return response.generations[0].text
```

---

## 🎯 Run the Pipeline

### 7️⃣ Complete Example:
```python
query = "What is Machine Learning?"
retrieved_docs = retrieve_documents(query)
answer = generate_answer(retrieved_docs, query)

print("Query:", query)
print("Answer:", answer)
```

---

## 📊 Example Outputs:
- **Q:** What is Machine Learning?  
  **A:** Machine Learning is a field focused on algorithms that enable computers to learn from data...

---

# 🧠 Interactive QA Bot with Document Upload 📄

## PART 2: Overview

The **Interactive QA Bot** allows users to upload PDFs, extract content, and ask questions. It uses **Pinecone** for retrieval and **Cohere** for generating answers.

<p align="center">
    <img src="Images/Screenshot 2024-09-20 230039.png" alt="BannerImg">
</p>

---

## Key Features
- ✨ **PDF Upload**  
- 🔍 **Document Retrieval**  
- 🤖 **AI-Powered Q&A**  
- 💻 **Interactive Interface**  
- 🚀 **Fast and Scalable**  

---

## Technology Stack
| **Tool**       | **Usage**                          |
| -------------- | ---------------------------------- |
| **Streamlit**  | User interface for PDFs and queries |
| **Pinecone**   | Vector database for embeddings |
| **Cohere**     | Language model for answers |
| **Transformers** | For embeddings |
| **PyPDF2**     | Extracting text from PDFs |

---

## 🚀 Quick Start

### Prerequisites
- **Python 3.10+**
- API keys for **Cohere** and **Pinecone**.

### 1. Clone the Repository
```bash
git clone https://github.com/yourusername/interactive-qa-bot.git
cd interactive-qa-bot
```

### 2. Set Up Virtual Environment
```bash
python -m venv venv
# Activate: Windows: venv\Scripts\activate | macOS/Linux: source venv/bin/activate
```

### 3. Install Dependencies
```bash
pip install -r requirements.txt
```

### 4. Configure API Keys
Update your API keys in **Back_End.py**.

### 5. Run the Application
```bash
streamlit run Front_End.py
```
Visit `http://localhost:8501`.

---

## 🛠 Project Structure
```
/project-directory
    ├── Front_End.py
    ├── Back_End.py
    ├── requirements.txt
    ├── Dockerfile
    └── README.md
```

---

## ⚙️ Backend Functions

### 1. **Embedding Text**
```python
def embed_text(text):
    # Embedding logic here
```

### 2. **Process PDF**
```python
def process_pdf(pdf_file):
    # Extract text from PDF
```

### 3. **Save Document Embeddings**
```python
def save_document_embeddings(text):
    # Store embeddings in Pinecone
```

---