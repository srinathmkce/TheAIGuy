"""Shared pipeline helpers for the RAG notebook series.

Every notebook from 01 onward loads `rag.pdf`, strips its running header/footer,
and (from notebook 3 onward) splits it into chapter-tagged chunks the same way.
This module centralizes that pipeline so each notebook can import it instead of
redefining the same regexes and loader/splitter calls.
"""

import os
import re
from pathlib import Path

from langchain_community.document_loaders import PyPDFLoader
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter

DEFAULT_CHUNK_SIZE = 800
DEFAULT_CHUNK_OVERLAP = 100
DEFAULT_EMBEDDING_MODEL = "sentence-transformers/paraphrase-MiniLM-L3-v2"  # ~17M params, 384-dim

HEADER_RE = re.compile(r"^RAG to Agentic RAG.*Course$", re.MULTILINE)
PAGE_NUM_RE = re.compile(r"^Page \d+$", re.MULTILINE)
CHAPTER_RE = re.compile(r"^CHAPTER (\d+)\n(.+)$", re.MULTILINE)


def maybe_colab_upload():
    """Open Colab's file picker if running in Colab; no-op (prints a note) otherwise."""
    try:
        from google.colab import files
        files.upload()  # -> saves to /content/<filename>
    except ImportError:
        print("Not running in Colab — skipping upload widget.")


def resolve_pdf_path(filename="rag.pdf"):
    """Prefer the Colab upload location; fall back to this repo's dataset/ folder locally."""
    colab_path = f"/content/{filename}"
    if os.path.exists(colab_path):
        return colab_path
    local_path = str(Path("../dataset") / filename)
    assert os.path.exists(local_path), f"Could not find {filename} - upload it first."
    return local_path


def clean_page(text: str) -> str:
    """Strip rag.pdf's repeated running header and page-number footer from one page."""
    text = HEADER_RE.sub("", text)
    text = PAGE_NUM_RE.sub("", text)
    return text.strip()


def load_pdf(pdf_path):
    """Load a PDF into one Document per page via PyPDFLoader."""
    return PyPDFLoader(pdf_path).load()


def load_clean_text(pdf_path):
    """Load a PDF and return (pages, full_text) with headers/footers stripped and pages joined."""
    pages = load_pdf(pdf_path)
    cleaned_pages = [clean_page(doc.page_content) for doc in pages]
    full_text = "\n\n".join(cleaned_pages)
    return pages, full_text


def split_into_chapters(text: str):
    """Return a list of {chapter_num, chapter_title, text} dicts, one per CHAPTER heading."""
    matches = list(CHAPTER_RE.finditer(text))
    chapters = []
    for i, m in enumerate(matches):
        start = m.end()
        end = matches[i + 1].start() if i + 1 < len(matches) else len(text)
        chapters.append({
            "chapter_num": m.group(1),
            "chapter_title": m.group(2).strip(),
            "text": text[start:end].strip(),
        })
    return chapters


def chunk_chapters(chapters, chunk_size=DEFAULT_CHUNK_SIZE, chunk_overlap=DEFAULT_CHUNK_OVERLAP, numeric_chapter_num=False):
    """Chapter-aware chunking: split *within* each chapter so no chunk crosses a chapter boundary.

    Set numeric_chapter_num=True to also attach a chapter_num_int field (needed for
    Chroma's numeric range filters, e.g. $gte/$lte — see notebook 6).
    """
    splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    chunks = []
    for c in chapters:
        for chunk in splitter.split_text(c["text"]):
            metadata = {"chapter_num": c["chapter_num"], "chapter_title": c["chapter_title"]}
            if numeric_chapter_num:
                metadata["chapter_num_int"] = int(c["chapter_num"])
            chunks.append(Document(page_content=chunk, metadata=metadata))
    return chunks


def load_chapter_chunks(pdf_path=None, chunk_size=DEFAULT_CHUNK_SIZE, chunk_overlap=DEFAULT_CHUNK_OVERLAP, numeric_chapter_num=False):
    """End-to-end pipeline: resolve path -> load+clean -> split into chapters -> chunk.

    Returns (pages, full_text, chapters, chunks).
    """
    if pdf_path is None:
        pdf_path = resolve_pdf_path()
    pages, full_text = load_clean_text(pdf_path)
    chapters = split_into_chapters(full_text)
    chunks = chunk_chapters(
        chapters, chunk_size=chunk_size, chunk_overlap=chunk_overlap, numeric_chapter_num=numeric_chapter_num
    )
    return pages, full_text, chapters, chunks


def get_embedder(model_name=DEFAULT_EMBEDDING_MODEL):
    """LangChain Embeddings wrapper around a sentence-transformers model."""
    from langchain_huggingface import HuggingFaceEmbeddings
    return HuggingFaceEmbeddings(model_name=model_name, encode_kwargs={"normalize_embeddings": True})


def build_chroma_store(chunks, embeddings=None, collection_name="rag_pdf_chapters"):
    """Embed chunks and index them into an in-memory Chroma collection."""
    os.environ.setdefault("ANONYMIZED_TELEMETRY", "FALSE")  # silence chromadb telemetry warnings
    from langchain_chroma import Chroma
    if embeddings is None:
        embeddings = get_embedder()
    return Chroma.from_documents(documents=chunks, embedding=embeddings, collection_name=collection_name)


# --- Retrieval metrics (derived in notebook 9, reused for the before/after comparison in notebook 10) ---

def precision_at_k(retrieved_ids, relevant_ids, k):
    top_k = retrieved_ids[:k]
    hits = sum(1 for doc_id in top_k if doc_id in relevant_ids)
    return hits / k


def recall_at_k(retrieved_ids, relevant_ids, k):
    top_k = retrieved_ids[:k]
    hits = sum(1 for doc_id in top_k if doc_id in relevant_ids)
    return hits / len(relevant_ids)


def reciprocal_rank(retrieved_ids, relevant_ids):
    for rank, doc_id in enumerate(retrieved_ids, start=1):
        if doc_id in relevant_ids:
            return 1.0 / rank
    return 0.0


def average_precision(retrieved_ids, relevant_ids):
    hits, running_precisions = 0, []
    for rank, doc_id in enumerate(retrieved_ids, start=1):
        if doc_id in relevant_ids:
            hits += 1
            running_precisions.append(hits / rank)
    return sum(running_precisions) / len(relevant_ids) if relevant_ids else 0.0


def ndcg_at_k(retrieved_ids, relevant_ids, k):
    import math

    def dcg(ids):
        return sum(1.0 / math.log2(rank + 1) for rank, doc_id in enumerate(ids, start=1) if doc_id in relevant_ids)

    actual = dcg(retrieved_ids[:k])
    ideal = dcg(list(relevant_ids)[:k])  # best possible ordering: every relevant doc ranked first
    return actual / ideal if ideal > 0 else 0.0


# 8 queries spanning 8 different chapters of rag.pdf, each paired with a short substring that
# appears in exactly one chunk of its answer - built once in notebook 9, reused in notebook 10.
EVAL_QUERIES = [
    ("What does the Okapi BM25 formula account for besides term frequency?", "Okapi Best Match 25"),
    ("How can giving a language model outside documents stop it from making things up?", "dynamic, external knowledge source"),
    ("What kind of queries is keyword search the strongest at handling?", "BM25 excels at: precise entity names"),
    ("Why is reciprocal rank fusion considered more robust than manually tuning weights?", "RRF is parameter-light and robust"),
    ("Which retrieval metric matters most for a RAG system with a fixed context window?", "Recall@K is the primary metric for RAG retrieval evaluation"),
    ("Why can't a bi-encoder capture fine-grained relevance between a query and a document?", "cannot consider the specific interaction between the query terms"),
    ("Give an example of two sentences that mean the same thing but share no words.", "The individual experienced angina"),
    ("When should a team pick RAG over fine-tuning a model?", "RAG is the right choice when your primary need is knowledge grounding"),
]


def build_eval_set(chunks):
    """Resolve EVAL_QUERIES' substrings against a chunked rag.pdf, returning [{"query", "relevant_idx"}, ...]."""
    def chunk_idx(unique_substring):
        return next(i for i, d in enumerate(chunks) if unique_substring in d.page_content)

    return [{"query": query, "relevant_idx": chunk_idx(substring)} for query, substring in EVAL_QUERIES]
