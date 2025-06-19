from helpers import read_pdf_text, extract_clauses

text = read_pdf_text("pcp.pdf")
clauses = extract_clauses(text)

print("10.0" in clauses)         # Should print True
print(clauses.get("10.0", "❌ Not found"))
