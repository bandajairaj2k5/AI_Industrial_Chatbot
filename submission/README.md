# AI-Powered Industrial Chatbot

An evidence-grounded Retrieval-Augmented Generation (RAG) assistant designed for industrial robotics, unmanned aerial vehicles (UAVs), avionics, and embedded engineering documentation.

---

## 📌 Project Overview

The **AI-Powered Industrial Chatbot** enables engineers, technicians, and students to query complex technical manuals using natural language. Rather than relying on generic, ungrounded language models that hallucinate critical engineering specifications, this system indexes proprietary PDF manuals, retrieves the exact relevant passages using dense vector embeddings, and generates verified answers cited with source filenames and page numbers.

To address computational and memory constraints on standard laptops (8 GB RAM), the application implements a **decoupled dual-mode architecture**:
* **Remote Cloud Mode (Default)**: Leverages Groq Cloud LPU acceleration with `qwen/qwen3.8-27b` to deliver sub-second responses (~0.75 s) while maintaining a lightweight client footprint (~177 MB RAM).
* **Local Offline Mode (Fallback)**: Preserves an offline CPU pipeline powered by Hugging Face Transformers (`Qwen/Qwen2.5-1.5B-Instruct`) for offline evaluation.

---

## 🚀 Key Features

* **Local Semantic Retrieval**: Dense vector indexing over 619 document chunks across six specialized technical PDFs.
* **Exact Mathematical Ranking**: FAISS `IndexFlatIP` performing cosine similarity ranking on 384-dimensional normalized vectors in under 50 milliseconds.
* **Evidence Grounding & Citation**: Every answer is accompanied by verified citations containing document filenames, page numbers, and cosine similarity scores.
* **Hallucination Resistance**: Strict prompt constraints compel the model to state clearly when information is not established by the technical documents.
* **Dual User Interfaces**:
  * **Streamlit Web Application (`app.py`)**: Full conversational interface with multi-turn memory and collapsible citation cards.
  * **Command-Line Interface (`chatbot.py`)**: Lightweight, dependency-minimal terminal chat loop.
* **Production-Grade Security**: Automatic `.env` loading with syntax validation, placeholder interception, and strict `.gitignore` exclusion.

---

## 🏗️ System Architecture

```
                                  [ Technical PDFs ]
                                          │
                                   ( ingest.py )
                                          │
                  ┌───────────────────────┴───────────────────────┐
                  ▼                                               ▼
         [ FAISS FlatIP Index ]                         [ Metadata Store ]
         ( industrial.index )                           ( metadata.json )
         ( 619 dense vectors )                          ( 619 text chunks )
                  │                                               │
                  └───────────────────────┬───────────────────────┘
                                          │
                                 [ User Technical Query ]
                                          │
                          ( all-MiniLM-L6-v2 Embedding )
                                          │
                          [ FAISS Top-K Search (k=3) ]
                                          │
                             [ Context Block Builder ]
                                          │
                  ┌───────────────────────┴───────────────────────┐
                  │                                               │
        IF INFERENCE_BACKEND="remote"                   IF INFERENCE_BACKEND="local"
                  ▼                                               ▼
     [ Groq Cloud LPU REST API ]                      [ Hugging Face Transformers ]
     Model: qwen/qwen3.8-27b                          Model: Qwen2.5-1.5B-Instruct
     Latency: ~0.75 s                                 Latency: ~30–60 s
     Client RAM: ~177 MB                              Client RAM: ~5,200 MB
                  │                                               │
                  └───────────────────────┬───────────────────────┘
                                          │
                               [ Grounded Response ]
                                          │
                        ┌─────────────────┴─────────────────┐
                        ▼                                   ▼
              [ Streamlit Web UI ]                 [ Interactive CLI ]
              ( http://localhost:8501 )            ( chatbot.py )
```

---

## 📁 Project Directory Structure

```text
AI_Industrial_Chatbot/
├── documents/                               # Source engineering PDF documents
│   ├── Embedded System Design...UAV.pdf
│   ├── products_and_solutions_for_drones.pdf
│   ├── ROBOTICS AND AUTOMATION-R22.pdf
│   ├── SECA4003.pdf
│   ├── lecture1424354515.pdf
│   └── Module5. (2).pdf
├── vectorstore/                             # Vector database artifacts
│   ├── industrial.index                     # Serialized FAISS IndexFlatIP (619 vectors)
│   └── metadata.json                        # Extracted chunk text and page metadata
├── report/                                  # Academic deliverables
│   ├── AI_Industrial_Chatbot_Report.pdf     # 14-page formal project report
│   ├── AI_Industrial_Chatbot_Report.md      # Editable markdown source
│   └── test_suite_results.json              # Benchmark measurements (Tests A-E)
├── Screenshots/                             # Demonstration evidence & guides
│   └── SCREENSHOT_CHECKLIST.md              # Screenshot capture procedures
├── app.py                                   # Streamlit web application
├── chatbot.py                               # Core RAG engine & CLI entry point
├── chatbot_backup.py                        # Original unedited baseline backup
├── ingest.py                                # PDF processing & indexing script
├── requirements.txt                         # Curated project dependencies
├── .env.example                             # Configuration template with placeholders
├── .gitignore                               # Secret and artifact exclusion rules
└── README.md                                # System documentation
```

---

## 🛠️ Prerequisites & Installation

### 1. Prerequisites
* **Operating System**: Windows 10/11, macOS, or Linux.
* **Python**: Python 3.10 to 3.14 (Verified on Python 3.14.7).
* **Hardware**: 8 GB RAM minimum for Remote mode; 16 GB+ recommended for Local CPU mode.

### 2. Setup Virtual Environment
Open PowerShell or your terminal in the project directory:

```powershell
# Create virtual environment
python -m venv .venv

# Activate virtual environment (Windows PowerShell)
.\.venv\Scripts\Activate.ps1

# (Alternative: Windows Command Prompt)
# .\.venv\Scripts\activate.bat
```

### 3. Install Dependencies
```powershell
pip install -r requirements.txt
```

---

## ⚙️ Environment Configuration

The application securely loads settings from a local `.env` file without exposing credentials.

1. **Create `.env` from template**:
   ```powershell
   Copy-Item .env.example .env
   ```
2. **Configure your settings**:
   Open `.env` in any text editor:
   ```env
   # Inference backend: "remote" (recommended) or "local"
   INFERENCE_BACKEND=remote

   # Groq API Key (obtain from https://console.groq.com/keys)
   GROQ_API_KEY=gsk_your_actual_key_here

   # Verified active Groq model ID
   GROQ_MODEL=qwen/qwen3.8-27b

   # Groq API endpoint
   GROQ_BASE_URL=https://api.groq.com/openai/v1
   ```
3. *Security Note*: `.env` is explicitly ignored by `.gitignore`. **Never commit `.env` or your API key to Git.**

---

## 🖥️ How to Run the Application

### Option A: Launch the Streamlit Web Interface (Recommended)
```powershell
.\.venv\Scripts\streamlit.exe run app.py
```
Open your browser to: **`http://localhost:8501`**

### Option B: Launch the Interactive CLI
```powershell
.\.venv\Scripts\python.exe chatbot.py
```
Type your question at the prompt. Enter `exit` or `quit` to close the session.

---

## 🔄 Document Re-Ingestion (Optional)

The project includes pre-built vectorstore files (`industrial.index` and `metadata.json`). If you add new PDFs to `documents/`:
```powershell
.\.venv\Scripts\python.exe ingest.py
```
*Note: This will re-chunk all documents and regenerate the FAISS index.*

---

## 📊 Empirical Test Results

The RAG engine was evaluated across eight standardized test cases using the live production Groq backend (`qwen/qwen3.8-27b`) and FAISS index:

| Test ID | Question Summary | Category | Sources | Latency | Grounding Verification |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **Test A** | Agricultural UAV main components | System Architecture | `Embedded System Design...UAV.pdf` (p. 2, 3) | **1.67 s** | **Supported**: Synthesizes redundant flight controls, 2.5mm frame, X11 Plus motors, and Pixhawk 6X. |
| **Test B** | Agricultural UAV motors | Exact Specifications | `Embedded System Design...UAV.pdf` (p. 3, 9, 2) | **1.64 s** | **Supported**: Extracts Hobbywing X11 Plus 85kv 11118 motors from Table 3.1. Zero hallucination. |
| **Test C** | Role of ESC in a drone | Functional Electronics | `products_and_solutions_for_drones.pdf` (p. 3)<br>`SECA4003.pdf` (p. 85, 38) | **2.85 s** | **Supported**: Explains motor speed control algorithms, efficiency, and battery extension. |
| **Test D** | Drone electronics curriculum | Corpus Coverage | `products_and_solutions_for_drones.pdf` (p. 1)<br>`SECA4003.pdf` (p. 1, 86) | **2.90 s** | **Supported**: Identifies SECA4003 syllabus modules (Units 1 & 5) and associated product solutions. |
| **Test E** | Definition of "electronics" | Conceptual + Grounded | `SECA4003.pdf` (p. 3, 86, 1) | **3.91 s** | **Dual Grounding**: Supplies standard engineering definition and grounds application in drone PCBA/hardware. |
| **Test F** | Coffee brewing temperature | Out of Domain | `[]` (Empty / Suppressed) | **0.01 s** | **Refusal Verified**: Intercepted by `RELEVANCE_THRESHOLD = 0.35`. Zero spurious citations. |
| **Test G** | Submersible pressure hull depth | Absent Specification | `[]` (Empty / Suppressed) | **0.01 s** | **Refusal Verified**: Intercepted by `RELEVANCE_THRESHOLD = 0.35`. Clean citation suppression. |
| **Test H** | Battery specs & recharge cycles | Partially Answerable | `Embedded System Design...UAV.pdf` (p. 6, 3, 2) | **2.22 s** | **Partial Grounding**: Details 12S 44.4V battery pack; explicitly notes recharge cycle limit is omitted. |

---


## ⚠️ Known Limitations

1. **Remote Mode Connectivity**: Remote inference requires outbound access to `api.groq.com`. If internet connectivity is interrupted, the chatbot displays a sanitized connection notice.
2. **Local Inference Hardware Overhead**: The local Hugging Face fallback (`Qwen2.5-1.5B-Instruct`) requires ~5 GB of system RAM. On laptops with 8 GB RAM, running local generation causes high memory usage and sluggish token generation.
3. **Pure Lexical vs Semantic Disconnects**: In rare instances where user queries use ambiguous pronouns (*"the UAV"* instead of *"the agricultural UAV"*), general classification slides may score higher than specific component tables.

---

## 🔧 Troubleshooting

| Issue | Cause | Resolution |
| :--- | :--- | :--- |
| **HTTP 401 Unauthorized** | Groq API key is missing or set to template placeholder. | Open `.env` and verify `GROQ_API_KEY=gsk_...` with your active key from Groq Console. |
| **HTTP 404 Model Not Found** | Outdated or deprecated model ID in configuration. | Set `GROQ_MODEL=qwen/qwen3.8-27b` or `openai/gpt-oss-20b` in `.env`. |
| **Streamlit Port In Use** | A previous Streamlit server is already bound to port 8501. | Run `streamlit run app.py --server.port 8502` to use an alternate port. |
| **High Memory / System Lag** | Local CPU mode was enabled on an 8 GB RAM machine. | Switch to `INFERENCE_BACKEND=remote` in `.env` or sidebar to free up ~5 GB of RAM. |

---

## 📄 Academic Project Details
* **Student Name**: [Student Name]
* **Roll Number**: [Roll Number]
* **Institution**: [College / University Name]
* **Supervisor**: [Project Supervisor Name]
* **Academic Year**: 2025 – 2026
