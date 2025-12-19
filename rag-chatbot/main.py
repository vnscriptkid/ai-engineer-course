# Import standard libraries for file handling and text processing
import os, pathlib, textwrap, glob

# Load documents from various sources (URLs, text files, PDFs)
from langchain_community.document_loaders import UnstructuredURLLoader, TextLoader, PyPDFLoader

# Split long texts into smaller, manageable chunks for embedding
from langchain.text_splitter import RecursiveCharacterTextSplitter

# Vector store to store and retrieve embeddings efficiently using FAISS
from langchain.vectorstores import FAISS

# Generate text embeddings using OpenAI or Hugging Face models
from langchain.embeddings import OpenAIEmbeddings, HuggingFaceEmbeddings, SentenceTransformerEmbeddings

# Use local LLMs (e.g., via Ollama) for response generation
from langchain.llms import Ollama

# Build a retrieval chain that combines a retriever, a prompt, and an LLM
from langchain.chains import ConversationalRetrievalChain

# Create prompts for the RAG system
from langchain.prompts import PromptTemplate

import streamlit as st
print("✅ Libraries imported! You're good to go!")

#############################################################
#############################################################
pdf_paths = glob.glob("data/Everstorm_*.pdf")
raw_docs = []

# --- Load PDFs (each page is a Document) ---
for path in pdf_paths:
    raw_docs.extend(PyPDFLoader(path).load())

print(f"Loaded {len(raw_docs)} PDF pages from {len(pdf_paths)} files.")

URLS = [
    "https://developer.bigcommerce.com/docs/store-operations/shipping",
    "https://developer.bigcommerce.com/docs/store-operations/orders/refunds",
]

try:
    loader = UnstructuredURLLoader(urls=URLS)
    web_raw_docs = loader.load()
    print(f"Fetched {len(web_raw_docs)} documents from the web.")

    # ✅ Correct: extend with a list of Documents
    raw_docs.extend(web_raw_docs)

except Exception as e:
    print("⚠️  Web fetch failed, using offline copies:", e)

print(f"Loaded {len(raw_docs)} documents total (pdf + web).")
#############################################################
#############################################################
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=300,
    chunk_overlap=30,
)

chunks = text_splitter.split_documents(raw_docs)

print(f"✅ {len(chunks)} chunks ready for embedding")
#############################################################
#############################################################
embeddings = SentenceTransformerEmbeddings(
    model_name="thenlper/gte-small"
)

vectordb = FAISS.from_documents(chunks, embeddings)
retriever = vectordb.as_retriever(search_kwargs={"k": 8})

vectordb.save_local("faiss_index")

print("✅ Vector store with", vectordb.index.ntotal, "embeddings")
#############################################################
#############################################################
SYSTEM_TEMPLATE = """
You are a Customer Support Chatbot.

You MUST answer using ONLY the information in <context>.
If the answer is not in <context>, reply exactly: "I'm not sure from the docs."

Rules:
1) If the answer IS in <context>, quote the exact sentence(s) that support it.
2) If the answer is NOT in <context>, reply exactly: "I'm not sure from the docs."
3) Keep the answer to 1-3 sentences.
4) Add citations like [source: ...] when available.

<context>
{context}
</context>

Question: {question}
Answer:
"""


prompt = PromptTemplate(template=SYSTEM_TEMPLATE, input_variables=["context", "question"])
llm = Ollama(model="gemma3:1b", temperature=0.1)
chain = ConversationalRetrievalChain.from_llm(llm, retriever, combine_docs_chain_kwargs={"prompt": prompt}, return_source_documents=True)
#############################################################
#############################################################
st.set_page_config(page_title="RAG Demo", page_icon="💬")

st.title("📄 RAG Chat Demo")
st.caption("Ask questions based on the indexed documents")

# --- session state ---
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

# --- input box ---
question = st.text_input("Ask a question")

if question:
    with st.spinner("Thinking..."):
        result = chain({
            "question": question,
            "chat_history": st.session_state.chat_history
        })

        answer = result["answer"]
        st.session_state.chat_history.append((question, answer))

# --- display chat ---
for q, a in st.session_state.chat_history[::-1]:
    st.markdown(f"**You:** {q}")
    st.markdown(f"**Assistant:** {a}")
    st.markdown("---")