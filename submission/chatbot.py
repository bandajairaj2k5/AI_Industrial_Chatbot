import os
import sys
import json
from pathlib import Path

import faiss
import numpy as np
import requests
from sentence_transformers import SentenceTransformer


# ============================================================
# CONFIGURATION & ENVIRONMENT SETUP
# ============================================================

BASE_DIR = Path(__file__).resolve().parent

def _is_valid_groq_key(key: str) -> bool:
    """
    Validate that an API key is non-empty, not a known placeholder,
    and adheres to Groq's official key format (starts with 'gsk_').
    Never logs or exposes the key.
    """
    if not key:
        return False
    cleaned = key.strip().strip("'\"")
    known_placeholders = {
        "your_groq_api_key_here",
        "your_actual_groq_api_key",
        "replace_with_my_real_key",
        "your_api_key_here",
        "your_key_here",
        "replace_with_your_key",
        "gsk_your_key_here",
    }
    if cleaned.lower() in known_placeholders or "<" in cleaned or ">" in cleaned:
        return False
    if not cleaned.startswith("gsk_"):
        return False
    if len(cleaned) < 20:
        return False
    return True


def _load_env_file(env_path: Path):
    """
    Safely load key-value pairs from a local .env file into os.environ.
    Preserves existing valid environment variables without overwriting.
    Does not display, log, or expose secret values.
    """
    if not env_path.exists():
        return
    try:
        with open(env_path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line or line.startswith("#") or "=" not in line:
                    continue
                key, val = line.split("=", 1)
                key = key.strip()
                if " #" in val:
                    val = val.split(" #", 1)[0]
                val = val.strip().strip('"').strip("'")
                
                # Update if not present, or if current os.environ has an empty/placeholder value
                current = os.environ.get(key, "").strip()
                if not current or not _is_valid_groq_key(current):
                    os.environ[key] = val
    except Exception:
        pass


# Initial load of local .env if present
_load_env_file(BASE_DIR / ".env")

VECTORSTORE_DIR = BASE_DIR / "vectorstore"
INDEX_PATH = VECTORSTORE_DIR / "industrial.index"
METADATA_PATH = VECTORSTORE_DIR / "metadata.json"

# Local embedding model (unchanged, runs locally on CPU)
EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"

# Retrieval settings
TOP_K = 3
MAX_CONTEXT_CHARS = 2500
RELEVANCE_THRESHOLD = 0.35

# Inference Backend Configuration
# "remote" = Fast remote inference via Groq (low RAM, fast response)
# "local"  = Hugging Face Transformers on CPU (assignment fallback)
INFERENCE_BACKEND = os.environ.get("INFERENCE_BACKEND", "remote").strip().lower()

# Local Hugging Face Configuration (fallback)
LOCAL_LLM_NAME = "Qwen/Qwen2.5-1.5B-Instruct"
LOCAL_MAX_INPUT_TOKENS = 1536
LOCAL_MAX_NEW_TOKENS = 100
LOCAL_CPU_THREADS = 4


# ============================================================
# INDUSTRIAL CHATBOT
# ============================================================

class IndustrialChatbot:

    def __init__(self, backend=None):

        # Re-check .env so dynamic edits to .env are picked up immediately
        _load_env_file(BASE_DIR / ".env")

        self.backend = (backend or os.environ.get("INFERENCE_BACKEND", "remote")).strip().lower()
        if self.backend not in {"remote", "local"}:
            print(f"Warning: Unknown backend '{self.backend}', defaulting to 'remote'.")
            self.backend = "remote"

        print("\nInitializing AI Industrial Chatbot...")
        print(f"Active Inference Backend: {self.backend.upper()}")
        print("-" * 50)

        # Check required vectorstore files
        if not INDEX_PATH.exists():
            raise FileNotFoundError(
                f"FAISS index not found: {INDEX_PATH}\n"
                "Run python ingest.py first."
            )

        if not METADATA_PATH.exists():
            raise FileNotFoundError(
                f"Metadata file not found: {METADATA_PATH}\n"
                "Run python ingest.py first."
            )

        # ----------------------------------------------------
        # 1. Load document index and metadata
        # ----------------------------------------------------
        print("[1/4] Loading document index and metadata...")

        self.index = faiss.read_index(str(INDEX_PATH))

        with open(METADATA_PATH, "r", encoding="utf-8") as file:
            self.records = json.load(file)

        if self.index.ntotal != len(self.records):
            raise ValueError(
                "The FAISS index and metadata are inconsistent. "
                "Run python ingest.py again."
            )

        if self.index.ntotal == 0:
            raise ValueError(
                "The document index is empty. "
                "Add PDFs and run python ingest.py."
            )

        print(f"      Loaded {self.index.ntotal} document chunks.")

        # ----------------------------------------------------
        # 2. Load embedding model
        # ----------------------------------------------------
        print(f"[2/4] Loading local embedding model ({EMBEDDING_MODEL})...")

        self.embedding_model = SentenceTransformer(
            EMBEDDING_MODEL,
            device="cpu"
        )

        # ----------------------------------------------------
        # 3. Configure language model backend
        # ----------------------------------------------------
        if self.backend == "remote":
            raw_key = os.environ.get("GROQ_API_KEY", "").strip() or os.environ.get("LLM_API_KEY", "").strip()
            self.is_key_valid = _is_valid_groq_key(raw_key)
            self.api_key = raw_key if self.is_key_valid else ""

            self.model_id = os.environ.get("GROQ_MODEL", "").strip() or os.environ.get("LLM_MODEL", "qwen/qwen3.8-27b").strip()
            self.base_url = os.environ.get("GROQ_BASE_URL", "").strip() or os.environ.get("LLM_BASE_URL", "https://api.groq.com/openai/v1").strip()
            self.timeout = float(os.environ.get("GROQ_TIMEOUT", "30.0"))
            self.max_tokens = int(os.environ.get("GROQ_MAX_TOKENS", "1024"))
            self.temperature = float(os.environ.get("GROQ_TEMPERATURE", "0.2"))

            print(f"[3/4] Configuring Groq remote inference ({self.model_id})...")
            if self.is_key_valid:
                print("      API key status: CONFIGURED (valid 'gsk_' format detected).")
            elif raw_key:
                print("      API key status: PLACEHOLDER DETECTED.")
                print("      Notice: GROQ_API_KEY in .env is still a template placeholder.")
                print("      Please edit .env to set your real Groq key (starts with 'gsk_').")
            else:
                print("      API key status: NOT DETECTED.")
                print("      Please set GROQ_API_KEY in .env or as an environment variable.")

            # Ensure local heavy model is NOT loaded into RAM
            self.tokenizer = None
            self.llm = None

        else:
            # Local fallback: Load Hugging Face Transformers model
            print(f"[3/4] Loading local Hugging Face model ({LOCAL_LLM_NAME})...")
            print("      Note: Local mode consumes higher RAM and CPU resources.")

            import torch
            from transformers import AutoTokenizer, AutoModelForCausalLM

            torch.set_num_threads(LOCAL_CPU_THREADS)

            self.tokenizer = AutoTokenizer.from_pretrained(LOCAL_LLM_NAME)
            self.llm = AutoModelForCausalLM.from_pretrained(
                LOCAL_LLM_NAME,
                dtype=torch.float32,
                low_cpu_mem_usage=True
            )
            self.llm.eval()

            if self.tokenizer.pad_token_id is None:
                self.tokenizer.pad_token = self.tokenizer.eos_token

        # ----------------------------------------------------
        # 4. Ready
        # ----------------------------------------------------
        print("[4/4] Initialization complete!")
        print("\nAI Industrial Chatbot is ready.")

    # ========================================================
    # GREETING HANDLER
    # ========================================================

    def handle_greeting(self, question):

        normalized = question.lower().strip("!.,? ")

        greetings = {
            "hi",
            "hello",
            "hey",
            "hello there",
            "hi there",
            "good morning",
            "good afternoon",
            "good evening"
        }

        if normalized in greetings:
            return {
                "answer": (
                    "Hello! I am your AI Industrial Chatbot.\n\n"
                    "I can help you explore your technical documents "
                    "about electronics, drones, robotics, automation, "
                    "and engineering.\n\n"
                    "What would you like to know?"
                ),
                "sources": []
            }

        return None

    # ========================================================
    # DOCUMENT RETRIEVAL
    # ========================================================

    def retrieve(self, question):

        question = question.strip()

        if not question:
            return []

        # Embed the question locally with SentenceTransformer
        question_embedding = self.embedding_model.encode(
            [question],
            normalize_embeddings=True,
            convert_to_numpy=True
        )

        question_embedding = np.asarray(
            question_embedding,
            dtype=np.float32
        )

        # Search for the closest document chunks in FAISS
        k = min(TOP_K, self.index.ntotal)

        scores, indices = self.index.search(
            question_embedding,
            k
        )

        results = []

        for score, position in zip(scores[0], indices[0]):

            if position < 0:
                continue

            record = self.records[int(position)]

            results.append({
                "score": float(score),
                "source": record["source"],
                "page": record["page"],
                "text": record["text"]
            })

        return results

    # ========================================================
    # BUILD DOCUMENT CONTEXT
    # ========================================================

    def build_context(self, results):

        context_parts = []
        remaining_chars = MAX_CONTEXT_CHARS

        for result in results:

            passage = (
                f"[Source: {result['source']}, "
                f"Page: {result['page']}]\n"
                f"{result['text']}"
            )

            if remaining_chars <= 0:
                break

            if len(passage) > remaining_chars:
                passage = passage[:remaining_chars]

            context_parts.append(passage)
            remaining_chars -= len(passage)

        return "\n\n".join(context_parts)

    # ========================================================
    # REMOTE GENERATION (GROQ API)
    # ========================================================

    def _generate_remote(self, messages):
        """Send retrieved context and query to Groq remote inference API."""

        if not self.is_key_valid or not self.api_key:
            return (
                "Error: Groq API key is not configured or is set to a placeholder.\n\n"
                "To fix this:\n"
                "1. Open the '.env' file in this project.\n"
                "2. Update the line:\n"
                "   GROQ_API_KEY=your_actual_groq_api_key\n"
                "   (Your key should start with 'gsk_')\n"
                "3. Save the file and try again.\n\n"
                "Alternatively, set INFERENCE_BACKEND=local to run offline via local Hugging Face model."
            )

        endpoint = f"{self.base_url.rstrip('/')}/chat/completions"

        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
            "User-Agent": "AI-Industrial-Chatbot/1.0"
        }

        payload = {
            "model": self.model_id,
            "messages": messages,
            "temperature": self.temperature,
            "max_tokens": self.max_tokens
        }

        print(f"Generating answer via Groq remote API ({self.model_id})...")

        try:
            response = requests.post(
                endpoint,
                headers=headers,
                json=payload,
                timeout=self.timeout
            )

            if response.status_code == 200:
                data = response.json()
                choices = data.get("choices", [])
                if choices and "message" in choices[0]:
                    content = choices[0]["message"].get("content", "").strip()
                    if content:
                        return content
                return "Groq API returned an empty completion response. Please try rephrasing your question."

            elif response.status_code == 401:
                return (
                    "Authentication failed (HTTP 401): The provided Groq API key was rejected by the Groq server.\n"
                    "Please verify that your key is active and correctly copied from https://console.groq.com/keys."
                )

            elif response.status_code == 404:
                return (
                    f"Model or endpoint error (HTTP 404): The model '{self.model_id}' was not found on Groq.\n"
                    "Please verify that GROQ_MODEL is set to an active Groq production model "
                    "(e.g., qwen/qwen3.8-27b or openai/gpt-oss-20b)."
                )

            elif response.status_code == 429:
                return (
                    "Groq rate limit reached (HTTP 429): You have temporarily exceeded Groq API limits.\n"
                    "Please wait a few seconds and try again."
                )

            else:
                try:
                    error_data = response.json()
                    error_msg = error_data.get("error", {}).get("message", response.text)
                except Exception:
                    error_msg = response.text
                return f"Groq API error (HTTP {response.status_code}): {error_msg}"

        except requests.exceptions.Timeout:
            return (
                f"Groq API request timed out after {self.timeout} seconds.\n"
                "Please check your internet connection or increase GROQ_TIMEOUT."
            )

        except requests.exceptions.ConnectionError:
            return (
                "Network connection error: Unable to connect to the Groq API endpoint.\n"
                "Please verify your internet connection and network settings."
            )

        except Exception as error:
            return f"Unexpected error during remote generation: {error}"

    # ========================================================
    # LOCAL GENERATION (HUGGING FACE TRANSFORMERS FALLBACK)
    # ========================================================

    def _generate_local(self, messages):
        """Generate answer locally using Hugging Face Transformers on CPU."""

        if self.llm is None or self.tokenizer is None:
            return "Local language model is not loaded. Ensure INFERENCE_BACKEND=local is set."

        import torch

        prompt = self.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True
        )

        inputs = self.tokenizer(
            prompt,
            return_tensors="pt",
            truncation=True,
            max_length=LOCAL_MAX_INPUT_TOKENS
        )

        print("Generating answer locally on CPU...")

        with torch.inference_mode():
            output = self.llm.generate(
                **inputs,
                max_new_tokens=LOCAL_MAX_NEW_TOKENS,
                do_sample=False,
                use_cache=True,
                pad_token_id=self.tokenizer.pad_token_id,
                eos_token_id=self.tokenizer.eos_token_id
            )

        input_length = inputs["input_ids"].shape[1]
        generated_tokens = output[0][input_length:]

        answer_text = self.tokenizer.decode(
            generated_tokens,
            skip_special_tokens=True
        ).strip()

        if not answer_text:
            answer_text = (
                "The language model did not generate an answer. "
                "Please try a shorter or more specific question."
            )

        return answer_text

    # ========================================================
    # GENERATE ANSWER
    # ========================================================

    def answer(self, question):

        question = question.strip()

        if not question:
            return {
                "answer": "Please enter a question.",
                "sources": []
            }

        # Greetings do not invoke any language model
        greeting_response = self.handle_greeting(question)

        if greeting_response is not None:
            return greeting_response

        # Retrieve supporting information from local FAISS
        print("\nSearching technical documents...")

        results = self.retrieve(question)

        # Check if any retrieved passage meets the minimum relevance floor
        if not results or results[0]["score"] < RELEVANCE_THRESHOLD:
            return {
                "answer": (
                    "The provided technical documents do not contain information "
                    "relevant to this question. The knowledge base is focused on "
                    "industrial UAV systems, drone electronics, robotics, avionics, "
                    "and embedded automation."
                ),
                "sources": []
            }

        context = self.build_context(results)

        # Construct concise, evidence-grounded messages adhering to the answering policy
        # Note: Only retrieved passages and the user's question are sent.
        # Entire PDF files are NEVER transmitted.
        messages = [
            {
                "role": "system",
                "content": (
                    "You are an AI Industrial Chatbot specializing in electronics, "
                    "drones, robotics, automation, and industrial engineering.\n\n"
                    "Follow these strict answering guidelines:\n"
                    "1. DOCUMENT-GROUNDED ANSWERS: For questions asking about system specifications, "
                    "components, architectures, measurements, or procedures, base your answer strictly "
                    "on the provided document context. Cite the relevant document and page numbers.\n"
                    "2. NO FABRICATION: Never invent specifications, component models, or claims not found "
                    "in the context.\n"
                    "3. PARTIALLY ANSWERABLE QUESTIONS: Answer the specific aspects that are supported by "
                    "the context, and explicitly clarify which requested details or specifications are missing "
                    "from the documents.\n"
                    "4. GENERAL DEFINITIONS & CONCEPTS: If asked to define a foundational engineering term or "
                    "concept (such as 'electronics', 'robotics', or 'drone') whose textbook definition is not "
                    "explicitly in the text: provide a concise, standard technical definition as general technical "
                    "knowledge, and then immediately state how the indexed documents apply or instantiate that concept "
                    "(citing the relevant document sections). Do not fabricate citations for the general definition.\n"
                    "5. ABSENT OR UNRELATED TOPICS: If the context contains insufficient evidence to answer a "
                    "factual question, or if the topic is absent from the documents, clearly state that the indexed "
                    "materials do not contain the requested information. Do not speculate."
                )
            },
            {
                "role": "user",
                "content": (
                    f"DOCUMENT CONTEXT:\n{context}\n\n"
                    f"QUESTION:\n{question}\n\n"
                    "Answer concisely in a few clear paragraphs."
                )
            }
        ]

        # Dispatch to the selected backend
        if self.backend == "remote":
            answer_text = self._generate_remote(messages)
        else:
            answer_text = self._generate_local(messages)

        # Package supporting source information (filtering out sub-threshold chunks)
        sources = []
        for result in results:
            if result["score"] >= RELEVANCE_THRESHOLD:
                sources.append({
                    "source": result["source"],
                    "page": result["page"],
                    "similarity": round(result["score"], 4)
                })

        return {
            "answer": answer_text,
            "sources": sources
        }



# ============================================================
# COMMAND-LINE INTERFACE
# ============================================================

def main():

    try:
        chatbot = IndustrialChatbot()

    except Exception as error:
        print("\nERROR: Chatbot initialization failed.")
        print(f"Details: {error}")
        return

    print("\n" + "=" * 60)
    print("          AI-POWERED INDUSTRIAL CHATBOT")
    print(f"          Backend: {chatbot.backend.upper()}")
    print("=" * 60)

    print("\nAsk questions about your technical documents.")
    print("Enter 'exit' or 'quit' to close the chatbot.")

    while True:

        try:
            question = input("\nYou: ").strip()

        except (KeyboardInterrupt, EOFError):
            print("\n\nChatbot closed.")
            break

        if question.lower() in {"exit", "quit"}:
            print("\nThank you for using the chatbot.")
            break

        if not question:
            continue

        try:
            response = chatbot.answer(question)

            print("\n" + "=" * 60)
            print("AI RESPONSE")
            print("=" * 60)

            print(response["answer"])

            print("\n" + "-" * 60)
            print("SOURCE DOCUMENTS")
            print("-" * 60)

            if response["sources"]:
                for source in response["sources"]:
                    print(f"\nDocument: {source['source']}")
                    print(f"Page: {source['page']}")
                    print(f"Similarity score: {source['similarity']}")
            else:
                print("No document sources used.")

            print("\n" + "=" * 60)

        except KeyboardInterrupt:
            print("\nGeneration interrupted.")

        except Exception as error:
            print(f"\nERROR while answering: {error}")


if __name__ == "__main__":
    main()