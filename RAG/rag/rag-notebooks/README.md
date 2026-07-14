# RAG Pipeline Series

A 13-notebook, hands-on course that builds a Retrieval-Augmented Generation (RAG) system from the ground up — one concept per notebook, every example run against one real document instead of toy data.

## Objective

Most RAG tutorials either wave their hands at retrieval ("just embed and search") or drown you in framework boilerplate before you understand what's actually happening underneath. This series does neither: it builds every core RAG component **from first principles first**, then shows the production-grade tool (LangChain, Chroma, LangGraph) that replaces the hand-rolled version — so you understand *why* the abstraction exists, not just how to call it.

Every notebook runs against the same source document: **`rag.pdf`** ("RAG to Agentic RAG — A Comprehensive Course", 15 chapters, 58 pages), found at [`../dataset/rag.pdf`](../dataset/rag.pdf) relative to this folder. Using one real, structured document throughout (rather than a different toy snippet per notebook) means later notebooks can directly compare their results against earlier ones — and, in a nice bit of self-reference, `rag.pdf`'s own chapters (BM25, retrieval evaluation, re-ranking, agentic RAG, ...) explain the exact concepts each notebook is building, so the notebooks routinely quote and validate against the source material's own worked examples.

The series has two arcs:
- **Notebooks 1–10** build and measure the *retrieval* half of RAG: getting a document in, chunking it well, representing it as sparse and dense vectors, storing/filtering/combining those representations, and rigorously measuring retrieval quality.
- **Notebooks 11–13** close the loop: turning retrieved chunks into a grounded LLM answer (augmentation + generation), then turning that into an **agent** that can decide for itself whether to consult the document, the live web, both, or neither — first as a hand-rolled loop, then as an explicit, stateful LangGraph.

## Notebook-by-notebook

### Part 1 — Retrieval fundamentals

| # | Notebook | What it covers |
|---|---|---|
| 1 | [01_data_loading.ipynb](01_data_loading.ipynb) | Loads `rag.pdf` with LangChain's `PyPDFLoader` (one `Document` per page), inspects sample pages/metadata, and builds a per-page character/word-count table with pandas to sanity-check the extraction. |
| 2 | [02_chunking_strategies.ipynb](02_chunking_strategies.ipynb) | Builds and compares six chunking strategies of increasing sophistication — naive fixed-length, `RecursiveCharacterTextSplitter`, token-based (tiktoken), chapter-aware, bullet-list-aware, and table-aware — against `rag.pdf`'s real prose, tables, and bulleted "Key Takeaways" blocks. Establishes the series' `chunk_size=800`/`chunk_overlap=100` baseline. |
| 3 | [03_keyword_embeddings.ipynb](03_keyword_embeddings.ipynb) | Builds sparse retrieval from scratch: **TF-IDF** (scikit-learn) and **BM25** (`rank_bm25`), then shows both fail on paraphrased queries that don't share exact vocabulary with the answer. |
| 4 | [04_vector_embeddings.ipynb](04_vector_embeddings.ipynb) | Introduces **dense embeddings** via a small `sentence-transformers` model, shows they fix the paraphrase failure from notebook 3, and visualizes chunk embeddings in 2D (PCA) clustering by chapter. |
| 5 | [05_vector_store_chroma.ipynb](05_vector_store_chroma.ipynb) | Replaces the manual NumPy/cosine-similarity search from notebook 4 with a real vector database, **Chroma**, and introduces the `.as_retriever()` interface used throughout the rest of the series. |
| 6 | [06_metadata_filtering.ipynb](06_metadata_filtering.ipynb) | Filters Chroma searches using `$eq`/`$ne`/`$in`/`$nin`, numeric range operators (`$gt`/`$gte`/`$lt`/`$lte`), and compound `$and`/`$or` filters — plus the recall/precision trade-off of scoping a search too narrowly. |
| 7 | [07_keyword_dense_retrieval.ipynb](07_keyword_dense_retrieval.ipynb) | Wraps BM25 (`BM25Retriever`) and the Chroma store behind LangChain's shared `Retriever` interface (`.invoke(query) -> List[Document]`), so sparse and dense retrieval become interchangeable/combinable. |
| 8 | [08_hybrid_search_rrf.ipynb](08_hybrid_search_rrf.ipynb) | Implements **Reciprocal Rank Fusion (RRF)** from scratch to merge BM25 + dense rankings, then reproduces it with LangChain's `EnsembleRetriever` (from `langchain-classic`), showing hybrid search recovers from either strategy's individual blind spots. |
| 9 | [09_retrieval_metrics.ipynb](09_retrieval_metrics.ipynb) | Implements **Precision@K, Recall@K, MRR, MAP, and NDCG@K** from their definitions, validates them against `rag.pdf` Chapter 7's own worked example, and scores BM25/dense/hybrid retrieval on an 8-query labeled eval set. |
| 10 | [10_reranking.ipynb](10_reranking.ipynb) | Adds a **cross-encoder** re-ranking stage (`cross-encoder/ms-marco-MiniLM-L-6-v2`) on top of first-stage retrieval, quantifying its effect (MRR/NDCG@5, before vs. after) on the same eval set from notebook 9. |

### Part 2 — Augmentation, generation, and agents

| # | Notebook | What it covers |
|---|---|---|
| 11 | [11_end_to_end_pipeline_gradio.ipynb](11_end_to_end_pipeline_gradio.ipynb) | Closes the retrieve → augment → generate loop: rebuilds the dense retriever, stuffs retrieved chunks into a prompt template, generates answers with **Gemini** (`langchain-google-genai`), and wraps the whole thing (`rag_answer()`) in a **Gradio** chat UI. |
| 12 | [12_agentic_rag_with_tools.ipynb](12_agentic_rag_with_tools.ipynb) | Turns retrieval into a *tool* rather than a fixed step: the model chooses per-question between `search_rag_document` (the notebook 1–10 retriever) and `web_search` (key-less DuckDuckGo search via `ddgs`), via a hand-written tool-calling agent loop. |
| 13 | [13_autonomous_agentic_rag_langgraph.ipynb](13_autonomous_agentic_rag_langgraph.ipynb) | Rebuilds notebook 12's agent as an explicit **LangGraph** graph (`agent`/`tools` nodes, a conditional edge, `MemorySaver` checkpointing), adding persistent multi-turn memory per conversation thread and a real chat UI (`gr.ChatInterface`). |

### Shared code: `rag_utils.py`

[rag_utils.py](rag_utils.py) centralizes the pipeline every notebook from 03 onward would otherwise duplicate: `resolve_pdf_path`/`load_pdf`/`load_clean_text` (loading + header/footer stripping), `split_into_chapters`/`chunk_chapters`/`load_chapter_chunks` (chapter-aware chunking), `get_embedder`/`build_chroma_store` (dense retrieval), the five retrieval-metric functions, and `EVAL_QUERIES`/`build_eval_set` (the shared 8-query eval set used by notebooks 9 and 10). Notebooks still redefine some logic inline where teaching *that logic* is the notebook's whole point (e.g. notebook 2's six chunking strategies) — only cross-notebook boilerplate was moved into this module.

## Prerequisites

- **Python 3.11+** — see [pyproject.toml](pyproject.toml). Dependencies are pinned via [uv.lock](uv.lock); install with [uv](https://docs.astral.sh/uv/) (`uv sync`) or `pip install -e .` if you prefer plain pip.
- **`rag.pdf`** must be present. Locally, it's already checked in at [`../dataset/rag.pdf`](../dataset/rag.pdf) (a sibling folder to this one) — every notebook resolves it automatically via `resolve_pdf_path()`. In Google Colab, each notebook opens a file-picker cell to upload it instead.
- **A `.env` file in this folder with `GOOGLE_API_KEY`** — required only for notebooks 11–13 (Gemini generation). Get a free key from [Google AI Studio](https://aistudio.google.com/apikey); the `gemini-3-flash-preview` model used in these notebooks is free-tier-eligible. `.env` is git-ignored, so your key won't be committed.
- **No key needed** for notebooks 1–10, or for the web-search tool in notebooks 12–13 (`ddgs` queries DuckDuckGo directly).
- All notebooks are written to run **either in Google Colab or locally** — each has its own `%pip install` setup cell and a Colab file-upload fallback (`maybe_colab_upload()`), so no notebook depends on a prior one's kernel state.

## Other notes

- **Run order matters conceptually, not mechanically.** Each notebook rebuilds everything it needs from scratch (loading, chunking, embedding, indexing), so any notebook can be opened and run standalone — but the material builds progressively, and later notebooks assume you've seen the ideas from earlier ones (e.g. notebook 8 assumes you understand the two retrievers from notebook 7). Going 1 → 13 in order is recommended for first-time viewers.
- **Everything is in-memory.** The Chroma store is rebuilt fresh in every notebook run (no `persist_directory`), so results are fully reproducible but nothing is cached to disk between runs.
- **Embeddings model:** the series intentionally uses a very small model, `sentence-transformers/paraphrase-MiniLM-L3-v2` (~17M params, 384-dim), so every notebook encodes almost instantly on CPU — good for following along, not a claim that it's the best embedding model available.
- **Chapter list of `rag.pdf`:** 01 Introduction to RAG · 02 Evolution of Retrieval · 03 Data Ingestion · 04 Embeddings · 05 Vector Databases & Indexing · 06 Retrieval Techniques · 07 Retrieval Evaluation · 08 Re-ranking · 09 Augmentation · 10 Generation · 11 Generation Evaluation · 12 RAG vs Fine-tuning · 13 RAG with Tools · 14 Agentic RAG · 15 Production Best Practices. Notebooks 1–10 and 12–13 map closely onto chapters 1–8 and 13–14; chapters 10–11 and 15 (pure generation, generation evaluation, and production practices) are touched on by notebook 11 but aren't a dedicated notebook.
