import os
import re
from flask import Flask, request, render_template, session, jsonify
from dotenv import load_dotenv
from helpers import *
import markdown
import openai

# Load environment variables
load_dotenv()

# Initialize Flask
app = Flask(__name__)
print("🧠 app.py has started")
app.secret_key = "secret123"
print("✅ app.py is running...")

# PDF processing
PDF_PATH = os.path.join(os.path.dirname(__file__), "pcp.pdf")
FULL_TEXT = read_pdf_text(PDF_PATH)
CHUNKS = split_into_chunks(FULL_TEXT)
CHUNK_INDEX, _ = build_index(CHUNKS)
CLAUSE_MAP = extract_clauses(FULL_TEXT)

from clause_titles import CLAUSE_TITLES  # External title mapping

@app.route("/", methods=["GET", "POST"])
def index():
    if "history" not in session:
        session["history"] = []
    return render_template("index.html", history=session["history"], suggestions=[])

@app.route("/ask", methods=["POST"])
def ask():
    if "history" not in session:
        session["history"] = []

    answer = ""
    suggestions = []

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
            if re.search(r"\b(print|show|display|get|full|exact)\b", question.lower()):
                answer = "<br><br>".join([
                    f"<b>{k}</b>:<br>{v.replace('/n', '<br>')}"
                    for k, v in sub_clauses.items()
                ])
            else:
                answer = explain_clause(sub_clauses, question)
        else:
            top_chunks = search_chunks(question, CHUNKS, CHUNK_INDEX, top_k=5)
            context = "\n\n".join(top_chunks)
            if not context.strip():
                context = keyword_search(FULL_TEXT, question) or "Not found."

            prompt = f"""
Answer the question using the full context below.
Return a detailed and complete response. If the answer has multiple points or details, use bullet points. 
Otherwise, return in clean paragraph format with necessary formatting (bold, lists, breaks).

Context:
\"\"\"
{context}
\"\"\"

Question: {question}
Answer (Markdown format):
"""
            response = openai.ChatCompletion.create(
                model="gpt-3.5-turbo",
                messages=[{"role": "user", "content": prompt}],
                temperature=0.3,
                max_tokens=2048
            )
            answer = markdown.markdown(response["choices"][0]["message"]["content"].strip())

        session["history"].append((question, answer))

    return jsonify({"answer": answer, "history": session["history"]})

@app.route("/clear", methods=["POST"])
def clear_history():
    session.pop("history", None)
    return jsonify({"status": "cleared"})

if __name__ == "__main__":
    import sys
    try:
        port = int(sys.argv[1]) if len(sys.argv) > 1 else 5000
        print(f"🚀 Starting chatbot on port {5002}...")
        app.run(host="127.0.0.1", port=5002, debug=True)
    except Exception as e:
        print(f"❌ Error while starting Flask app: {e}")
