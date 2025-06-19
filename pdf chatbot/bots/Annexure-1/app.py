import os
import re
from flask import Flask, request, render_template, session
from dotenv import load_dotenv
from helpers import *

load_dotenv()
app = Flask(__name__)
app.secret_key = "secret123"

PDF_PATH = "Annexure-1.pdf"
FULL_TEXT = read_pdf_text(PDF_PATH)
CHUNKS = split_into_chunks(FULL_TEXT)
CHUNK_INDEX, _ = build_index(CHUNKS)
CLAUSE_MAP = extract_clauses(FULL_TEXT)
CLAUSE_TITLES = extract_clause_titles(FULL_TEXT)

@app.route("/", methods=["GET", "POST"])
def index():
    if "history" not in session:
        session["history"] = []

    answer = ""
    suggestions = []

    if request.method == "POST":
        if "summarize" in request.form:
            answer = summarize_text(FULL_TEXT)
            session["history"].append(("Summarize the document", answer))
        else:
            question = request.form["question"]

            if not re.search(r"[a-zA-Z]", question):
                question = translate_to_english(question)

            question = correct_grammar_and_spelling(question)

            regex_match = re.search(r"\b(\d+(?:\.\d+)*)\b", question)
            clause_key = regex_match.group(1) if regex_match else None
            if clause_key and '.' not in clause_key:
                clause_key += ".0"

            if not clause_key or clause_key not in CLAUSE_MAP:
                clause_key = match_clause_by_title(question, CLAUSE_TITLES, CLAUSE_MAP)

            if clause_key and clause_key in CLAUSE_MAP:
                sub_clauses = get_clause_family(CLAUSE_MAP, clause_key)

                if re.search(r"\b(print|show|display)\b", question.lower()):
                    answer = "<br><br>".join([f"<b>{k}</b>: {v}" for k, v in sub_clauses.items()])
                else:
                    answer = explain_clause(sub_clauses, question)
            else:
                top_chunks = search_chunks(question, CHUNKS, CHUNK_INDEX, top_k=5)
                context = "\n\n".join(top_chunks)
                if not context.strip():
                    context = keyword_search(FULL_TEXT, question) or "Not found."

                prompt = f"""
Answer the question using the context below.
Be specific and cover all important points.

Context:
\"\"\"
{context}
\"\"\"

Question: {question}
Answer (Markdown format):
"""
                response = openai.ChatCompletion.create(  # Updated
                    model="gpt-3.5-turbo",
                    messages=[{"role": "user", "content": prompt}],
                    temperature=0.3
                )
                answer = markdown.markdown(response["choices"][0]["message"]["content"].strip())

            session["history"].append((question, answer))

    return render_template("index.html", history=session["history"], suggestions=suggestions)

if __name__ == "__main__":
    import sys
    try:
        port = int(sys.argv[1]) if len(sys.argv) > 1 else 5000
        print(f"🚀 Starting chatbot on port {5001}...")
        app.run(host="127.0.0.1", port=5001, debug=True)
    except Exception as e:
        print(f"❌ Error starting Flask app: {e}")
