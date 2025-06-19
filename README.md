# SAIL Policy Chatbot 🤖
*AI-Powered Document Assistant for Steel Authority of India Limited*

A Flask-based chatbot that helps SAIL employees navigate complex policy documents  through natural language queries, clause lookup, and document summarization. Developed during Summer 2025 internship at SAIL.

## 🌟 Key Features
- **Clause Navigation**: Retrieve policy clauses using numbers or descriptive titles
- **Document Summarization**: Generate concise summaries of lengthy policy documents
- **Multilingual Support**: Auto-translates non-English queries to English
- **Contextual Q&A**: GPT-3.5 powered answers based on document context
- **PDF OCR**: Text extraction from scanned documents using Tesseract
- **Conversation History**: Maintains session-based chat history

## 🛠️ Technology Stack
### Core Components
| Component | Technology |
|-----------|------------|
| Backend | Python Flask |
| AI Engine | OpenAI GPT-3.5 Turbo |
| PDF Processing | PyPDF2, pdf2image |
| OCR | Tesseract |
| Vector Search | FAISS |
| NLP Toolkit | NLTK, RapidFuzz |

### Key Libraries

flask==2.2.5
openai==0.28.0
pytesseract==0.3.13
pdf2image==1.17.0
faiss-cpu==1.11.0
nltk==3.9.1
rapidfuzz==3.13.0


## 🚀 Installation
1. Clone repository:

git clone https://github.com/AryanPandey-pvt/AI-PDF-Chatbot.git
cd AI-PDF-Chatbot


2. Install dependencies:

pip install -r requirements.txt


3. Set OpenAI API key:

export OPENAI_API_KEY='your-api-key'


4. Run the application:

python app.py

python controller.py(for pdf launchers)

python app.py(for pdf chatbots)


## 🧠 How It Works
1. **Document Processing**:
   - PDFs are converted to text using OCR (Tesseract)
   - Policy clauses are extracted with regex pattern matching
   - Document is chunked and indexed using FAISS

2. **Query Handling**:
if clause_key in CLAUSE_MAP:
return clause_explanation
else:
return gpt_contextual_answer


3. **Special Features**:
- Automatic grammar correction
- Clause family retrieval (e.g., "5.4" returns all 5.4.x clauses)
- Multi-format output (raw text or explained summaries)

## 📂 Project Structure
sail-chatbot/
├── app.py # Main application
├── helpers.py # PDF processing & AI functions
├── clause_titles.py # Clause title mappings
├── templates/
│ └── index.html # Chat interface
├── data/
│ └── Annexure-1.pdf # Policy document
├── clause_keys_debug.txt # Debugging keys
└── requirements.txt # Dependencies


## 💡 Usage Examples
1. **Clause Retrieval**:
   > "Show clause 5.4.2"
   
2. **Contextual Query**:
   > "Explain the empanelment process for retired officers"

3. **Document Summary**:
   > "Summarize the policy guidelines"

## 🌟 Internship Contribution
Developed during Summer 2025 internship at **Steel Authority of India Limited (SAIL)** under the guidance of SAIL's Engineering and Technology team. Special thanks to my mentor Mrs. S. Selvi for invaluable support.

---

*"Simplifying policy navigation for India's steel industry"*




