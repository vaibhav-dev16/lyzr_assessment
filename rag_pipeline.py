import json
import numpy as np
import os
from dotenv import load_dotenv

from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_openai import ChatOpenAI
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser

load_dotenv()
DEEPSEEK_KEY = os.getenv("DEEPSEEK_API_KEY")

embedding_model = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)

def load_vector_db():
    with open("clinic_info.json", "r") as f:
        data = json.load(f)

    texts = [f"Q: {item['question']}\nA: {item['answer']}" for item in data]

    vectordb = Chroma.from_texts(
        texts=texts,
        embedding=embedding_model,
        collection_name="clinic_faq",
        persist_directory="./chroma_store"
    )

    return vectordb

vectordb = load_vector_db()
retriever = vectordb.as_retriever(search_kwargs={"k": 3})

prompt = PromptTemplate.from_template("""
You are a clinic FAQ assistant. Answer ONLY using the context. Do NOT hallucinate.

Context:
{context}

Question: {question}

Answer clearly:
""")

llm = ChatOpenAI(
    model="deepseek-chat",
    api_key=DEEPSEEK_KEY,
    base_url="https://api.deepseek.com",
    temperature=0
)

def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)

qa_chain = (
    {
        "context": retriever | format_docs,
        "question": RunnablePassthrough()
    }
    | prompt
    | llm
    | StrOutputParser()
)

def cosine_similarity(a, b):
    a, b = np.array(a), np.array(b)
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b)))

last_question = None
last_embedding = None

def answer_question(user_q: str):
    global last_question, last_embedding

    q_emb = embedding_model.embed_query(user_q)

    if last_embedding is not None:
        sim = cosine_similarity(q_emb, last_embedding)
        is_followup = sim > 0.75
        final_q = f"{last_question}. {user_q}" if is_followup else user_q
    else:
        is_followup = False
        final_q = user_q

    answer = qa_chain.invoke(final_q)

    last_question = user_q
    last_embedding = q_emb

    return answer, is_followup
