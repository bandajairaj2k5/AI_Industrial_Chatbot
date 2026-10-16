# Demonstration Screenshots Capture Guide & Checklist

**Project**: AI-Powered Industrial Chatbot  
**Destination Folder**: `Screenshots/`  
**Purpose**: Visual demonstration evidence for academic evaluation, final report exhibits, and project submission.

---

### Security Pre-Check:
* ⚠️ **Never include credentials or API keys in screenshots.**
* Ensure the `.env` file, browser address bar tokens, or terminal environment variable assignments are not visible in any captured area.

---

### Screenshot Specifications & Checklist

| # | Filename | Application State / Screen | Exact Query / Action | What to Capture in Frame |
| :--- | :--- | :--- | :--- | :--- |
| **1** | `01_main_interface.png` | **Main Landing View** | Open `http://localhost:8501` | Full browser window showing the header, sidebar with 619 chunks / MiniLM model, and initial welcome greeting. |
| **2** | `02_drone_answer.png` | **Subsystems Response** | Enter: *"What are the main subsystems of the industrial drone?"* | The user message bubble and the assistant response detailing hardware, software, and mechanical subsystems. |
| **3** | `03_source_citations.png` | **Expanded Citations** | Click the `📑 Supporting Document Sources` accordion under Test A | Expanded card showing `SECA4003.pdf` Pages 3 & 6, `products_and_solutions_for_drones.pdf` Page 1, and cosine similarity badges. |
| **4** | `04_uav_specifications.png` | **Hardware Specifications** | Enter: *"What motors and frame material are specified for the heavy-lift agricultural UAV?"* | The response specifying Hobbywing X11 Plus motors and 2.5mm Galvanized Steel frame with source citations. |
| **5** | `05_missing_information.png` | **Refusal / Hallucination Test** | Enter: *"What is the maximum underwater diving depth of a submarine?"* | Assistant transparently stating that the technical manuals contain no submarine diving information. |
| **6** | `06_cli_demo.png` | **Interactive CLI Terminal** | Run `.\.venv\Scripts\python.exe chatbot.py` in PowerShell | Terminal output showing initialization, query processing, AI response, and the `SOURCE DOCUMENTS` table. |

---

### How to Capture & Save on Windows:
1. Open the Chatbot UI in your browser: **[http://localhost:8501](http://localhost:8501)**.
2. Press **`Win + Shift + S`** (Windows Snipping Tool).
3. Select the browser region or active terminal window.
4. Save the screenshot directly into `C:\Users\banaj\OneDrive\Desktop\AI_Industrial_Chatbot\Screenshots\` using the exact filenames listed above.
