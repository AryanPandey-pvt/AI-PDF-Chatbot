import os
import re
import faiss
import numpy as np
import tiktoken
import pytesseract
from PyPDF2 import PdfReader
from pdf2image import convert_from_path
import openai
from dotenv import load_dotenv
from difflib import get_close_matches
import markdown
from PIL import Image

load_dotenv()
openai.api_key = os.getenv("OPENAI_API_KEY")
tokenizer = tiktoken.get_encoding("cl100k_base")
CHUNK_SIZE = 300

def read_pdf_text(path):
    try:
        reader = PdfReader(path)
        text = "\n".join([page.extract_text() or '' for page in reader.pages])
        if text.strip():
            return text
    except Exception:
        pass
    print(f"[OCR] Falling back to Tesseract for: {os.path.basename(path)}")
    return read_pdf_with_ocr(path)

def read_pdf_with_ocr(pdf_path):
    poppler_path = r"C:\poppler\Library\bin"
    images = convert_from_path(pdf_path, dpi=300, poppler_path=poppler_path)
    text = ""
    for img in images:
        text += pytesseract.image_to_string(img) + "\n\n"
    return text

def split_into_chunks(text):
    words = text.split()
    chunks, chunk = [], []
    for word in words:
        chunk.append(word)
        if len(tokenizer.encode(" ".join(chunk))) > CHUNK_SIZE:
            chunks.append(" ".join(chunk))
            chunk = []
    if chunk:
        chunks.append(" ".join(chunk))
    return chunks

def embed_text(texts):
    if isinstance(texts, str): texts = [texts]
    response = openai.Embedding.create(  # Updated for old API
        input=texts,
        model="text-embedding-ada-002"
    )
    return np.array([d["embedding"] for d in response["data"]], dtype=np.float32)

def build_index(chunks):
    vectors = embed_text(chunks)
    dim = vectors.shape[1]
    index = faiss.IndexFlatL2(dim)
    index.add(vectors)
    return index, vectors

def extract_clauses(text):
    lines = text.splitlines()
    clauses = {}
    current_key, current_text = None, []
    for line in lines:
        match = re.match(r"^(\d+(?:\.\d+)*)(?:\s+|$)", line.strip())
        if match:
            if current_key and current_text:
                clauses[current_key] = " ".join(current_text)
            current_key = match.group(1)
            if '.' not in current_key:
                current_key += ".0"
            current_text = [line]
        elif current_key:
            current_text.append(line)
    if current_key and current_text:
        clauses[current_key] = " ".join(current_text)
    return clauses

def extract_clause_titles(text):
    titles = {}
    lines = text.splitlines()
    for line in lines:
        match = re.match(r"^(\d+(?:\.\d+)*)(?:\s+)(.+)", line.strip())
        if match:
            key = match.group(1)
            if '.' not in key:
                key += ".0"
            titles[key] = match.group(2).strip()
    return titles

def get_clause_family(clauses, key_prefix):
    return {k: v for k, v in clauses.items() if k.startswith(key_prefix)}

def search_chunks(query, chunks, index, top_k=5):
    q_vector = embed_text(query)[0].reshape(1, -1)
    D, I = index.search(q_vector, top_k)
    return [chunks[i] for i in I[0]]

def fuzzy_clause_match(clauses, query):
    candidates = list(clauses.keys())
    matches = get_close_matches(query, candidates, n=1, cutoff=0.6)
    return matches[0] if matches else None

def keyword_search(text, query):
    lines = text.lower().splitlines()
    results = [line for line in lines if any(q in line for q in query.lower().split())]
    return "\n".join(results[:3]) if results else None

def translate_to_english(question):
    response = openai.ChatCompletion.create(  # Updated
        model="gpt-3.5-turbo",
        messages=[{"role": "user", "content": f"Translate to English: {question}"}],
        temperature=0
    )
    return response["choices"][0]["message"]["content"].strip()

def correct_grammar_and_spelling(question):
    response = openai.ChatCompletion.create(  # Updated
        model="gpt-3.5-turbo",
        messages=[{"role": "user", "content": f"Correct grammar/spelling:\n{question}"}],
        temperature=0
    )
    return response["choices"][0]["message"]["content"].strip()

def summarize_text(text):
    prompt = f"Summarize:\n\n{text[:4000]}"
    response = openai.ChatCompletion.create(  # Updated
        model="gpt-3.5-turbo",
        messages=[{"role": "user", "content": prompt}],
        temperature=0.3
    )
    return markdown.markdown(response["choices"][0]["message"]["content"].strip())

def explain_clause(clause_dict, user_question):
    text = "\n".join([f"{k} {v}" for k, v in clause_dict.items()])
    prompt = f"""
You are a legal assistant. Based on the clause content below, answer the user's question clearly.
Use bullet points or paragraphs as needed. Be accurate and concise.

Clause:
\"\"\"
{text}
\"\"\"

Question: {user_question}
Answer (Markdown format):
"""
    response = openai.ChatCompletion.create(  # Updated
        model="gpt-3.5-turbo",
        messages=[{"role": "user", "content": prompt}],
        temperature=0.3
    )
    return markdown.markdown(response["choices"][0]["message"]["content"].strip())

def match_clause_by_title(query, clause_titles, clause_map):
    query_lower = query.lower()
    for key, title in clause_titles.items():
        if query_lower in title.lower() or title.lower() in query_lower:
            if key in clause_map:
                return key
    return None
