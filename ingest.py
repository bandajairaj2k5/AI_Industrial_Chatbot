from pathlib import Path
import json

import pymupdf
import faiss
import numpy as np
from sentence_transformers import SentenceTransformer


# Project directories
BASE_DIR = Path(__file__).resolve().parent
DOCUMENTS_DIR = BASE_DIR / "documents"
VECTORSTORE_DIR = BASE_DIR / "vectorstore"

INDEX_PATH = VECTORSTORE_DIR / "industrial.index"
METADATA_PATH = VECTORSTORE_DIR / "metadata.json"

# Embedding model
EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"

# Chunk settings
CHUNK_SIZE = 1200
CHUNK_OVERLAP = 200


def extract_pdf_text(pdf_path):
    """Extract text from a PDF document."""

    extracted_pages = []

    with pymupdf.open(pdf_path) as document:
        for page_number, page in enumerate(document, start=1):
            text = page.get_text("text").strip()

            if text:
                extracted_pages.append({
                    "page": page_number,
                    "text": text
                })

    return extracted_pages


def split_text(text):
    """Split text into overlapping character-based chunks."""

    chunks = []
    start = 0

    while start < len(text):
        end = min(start + CHUNK_SIZE, len(text))
        chunk = text[start:end].strip()

        if chunk:
            chunks.append(chunk)

        if end == len(text):
            break

        start = end - CHUNK_OVERLAP

    return chunks


def main():
    VECTORSTORE_DIR.mkdir(parents=True, exist_ok=True)

    pdf_files = sorted(DOCUMENTS_DIR.glob("*.pdf"))

    if not pdf_files:
        print(f"No PDF documents found in: {DOCUMENTS_DIR}")
        print("Place at least one industrial PDF in the documents folder.")
        return

    records = []

    print(f"Found {len(pdf_files)} PDF document(s).")

    for pdf_path in pdf_files:
        print(f"Processing: {pdf_path.name}")

        try:
            pages = extract_pdf_text(pdf_path)
        except Exception as error:
            print(f"Could not read {pdf_path.name}: {error}")
            continue

        for page_data in pages:
            chunks = split_text(page_data["text"])

            for chunk_number, chunk in enumerate(chunks, start=1):
                records.append({
                    "source": pdf_path.name,
                    "page": page_data["page"],
                    "chunk": chunk_number,
                    "text": chunk
                })

    if not records:
        print("No readable text was extracted from the PDFs.")
        print("Scanned PDFs may require OCR before processing.")
        return

    print(f"Created {len(records)} text chunks.")

    print(f"Loading embedding model: {EMBEDDING_MODEL}")

    model = SentenceTransformer(EMBEDDING_MODEL)

    texts = [record["text"] for record in records]

    embeddings = model.encode(
        texts,
        batch_size=16,
        show_progress_bar=True,
        normalize_embeddings=True,
        convert_to_numpy=True
    )

    embeddings = np.asarray(embeddings, dtype=np.float32)

    # Normalized embeddings allow inner-product search
    # to rank results by cosine similarity.
    index = faiss.IndexFlatIP(embeddings.shape[1])
    index.add(embeddings)

    faiss.write_index(index, str(INDEX_PATH))

    with open(METADATA_PATH, "w", encoding="utf-8") as file:
        json.dump(records, file, ensure_ascii=False, indent=2)

    print("\nDocument ingestion completed successfully.")
    print(f"Total chunks: {len(records)}")
    print(f"FAISS index: {INDEX_PATH}")
    print(f"Metadata: {METADATA_PATH}")


if __name__ == "__main__":
    main()