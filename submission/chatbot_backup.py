import json
from pathlib import Path

import faiss
import numpy as np
import torch

from sentence_transformers import SentenceTransformer
from transformers import AutoTokenizer, AutoModelForCausalLM


# ============================================================
# CONFIGURATION
# ============================================================

BASE_DIR = Path(__file__).resolve().parent

VECTORSTORE_DIR = BASE_DIR / "vectorstore"

INDEX_PATH = VECTORSTORE_DIR / "industrial.index"
METADATA_PATH = VECTORSTORE_DIR / "metadata.json"

EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
LLM_NAME = "Qwen/Qwen2.5-1.5B-Instruct"

# Performance settings for a CPU-based laptop
TOP_K = 3
MAX_CONTEXT_CHARS = 2500
MAX_INPUT_TOKENS = 1536
MAX_NEW_TOKENS = 100

# Limit CPU thread usage to avoid excessive contention.
CPU_THREADS = 4

torch.set_num_threads(CPU_THREADS)


# ============================================================
# INDUSTRIAL CHATBOT
# ============================================================

class IndustrialChatbot:

    def __init__(self):

        print("\nInitializing AI Industrial Chatbot...\n")

        # Check required files
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

        print("[1/4] Loading document index...")

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

        # ----------------------------------------------------
        # 2. Load embedding model
        # ----------------------------------------------------

        print("[2/4] Loading embedding model...")

        self.embedding_model = SentenceTransformer(
            EMBEDDING_MODEL,
            device="cpu"
        )

        # ----------------------------------------------------
        # 3. Load language model
        # ----------------------------------------------------

        print("[3/4] Loading Hugging Face language model...")

        self.tokenizer = AutoTokenizer.from_pretrained(
            LLM_NAME
        )

        self.llm = AutoModelForCausalLM.from_pretrained(
            LLM_NAME,
            dtype=torch.float32,
            low_cpu_mem_usage=True
        )

        self.llm.eval()

        # Ensure padding is configured.
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

        # Embed the question.
        question_embedding = self.embedding_model.encode(
            [question],
            normalize_embeddings=True,
            convert_to_numpy=True
        )

        question_embedding = np.asarray(
            question_embedding,
            dtype=np.float32
        )

        # Search for the closest document chunks.
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
    # GENERATE ANSWER
    # ========================================================

    def answer(self, question):

        question = question.strip()

        if not question:
            return {
                "answer": "Please enter a question.",
                "sources": []
            }

        # Greetings should not invoke the language model.
        greeting_response = self.handle_greeting(question)

        if greeting_response is not None:
            return greeting_response

        # Retrieve supporting information.
        print("\nSearching technical documents...")

        results = self.retrieve(question)

        if not results:
            return {
                "answer": (
                    "I could not find relevant information "
                    "in the available documents."
                ),
                "sources": []
            }

        context = self.build_context(results)

        # Construct a concise, evidence-grounded prompt.
        messages = [
            {
                "role": "system",
                "content": (
                    "You are an AI Industrial Chatbot specializing "
                    "in electronics, drones, robotics, automation, "
                    "and engineering.\n\n"
                    "Instructions:\n"
                    "- Give concise, clear answers.\n"
                    "- Use the provided document context as evidence.\n"
                    "- Do not invent specifications, measurements, "
                    "or document contents.\n"
                    "- If the context does not establish an answer, "
                    "say so clearly.\n"
                    "- Distinguish general technical knowledge "
                    "from facts supported by the supplied documents."
                )
            },
            {
                "role": "user",
                "content": (
                    f"DOCUMENT CONTEXT:\n{context}\n\n"
                    f"QUESTION:\n{question}\n\n"
                    "Answer in a maximum of a few short paragraphs."
                )
            }
        ]

        prompt = self.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True
        )

        # Tokenization can truncate some document context.
        inputs = self.tokenizer(
            prompt,
            return_tensors="pt",
            truncation=True,
            max_length=MAX_INPUT_TOKENS
        )

        print("Generating answer on CPU...")

        # Inference mode reduces memory overhead.
        with torch.inference_mode():

            output = self.llm.generate(
                **inputs,
                max_new_tokens=MAX_NEW_TOKENS,
                do_sample=False,
                use_cache=True,
                pad_token_id=self.tokenizer.pad_token_id,
                eos_token_id=self.tokenizer.eos_token_id
            )

        # Decode only the newly generated answer.
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

        # Return source information for transparency.
        sources = []

        for result in results:

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
                    print(
                        "Similarity score: "
                        f"{source['similarity']}"
                    )

            else:
                print("No document sources used.")

            print("\n" + "=" * 60)

        except KeyboardInterrupt:

            print("\nGeneration interrupted.")

        except Exception as error:

            print(f"\nERROR while answering: {error}")


if __name__ == "__main__":
    main()
    