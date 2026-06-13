import json
import time
import numpy as np
import kagglehub
from sie_sdk import SIEClient
from sie_sdk.types import Item
from tqdm import tqdm

# sie-server serve --host 0.0.0.0 --port 8080

MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
NUM_DOCUMENTS = 100000
BATCH_SIZE = 128
SIE_URL = "http://localhost:8080"


def load_articles(file_path: str, num_documents: int) -> list[dict]:
    """Load articles from JSONL format (one JSON object per line)."""
    articles = []
    with open(file_path, "r", encoding="utf-8") as f:
        for i, line in enumerate(f):
            if i >= num_documents:
                break
            if line.strip():
                articles.append(json.loads(line))
    return articles


def embed_abstracts(
    articles: list[dict],
    model_name: str = MODEL_NAME,
    batch_size: int = BATCH_SIZE,
    sie_url: str = SIE_URL,
) -> np.ndarray:
    client = SIEClient(sie_url)

    abstracts = [article.get("abstract", "") for article in articles]
    batches = [abstracts[i : i + batch_size] for i in range(0, len(abstracts), batch_size)]

    all_embeddings = []
    for batch in tqdm(batches, desc="Embedding abstracts", unit="batch"):
        items = [Item(text=text) for text in batch]
        results = client.encode(model_name, items)
        batch_embeddings = np.vstack([r["dense"] for r in results])
        all_embeddings.append(batch_embeddings)

    return np.vstack(all_embeddings)


def main():
    path = kagglehub.dataset_download("Cornell-University/arxiv")
    filtered_file = path + "/arxiv-metadata-oai-snapshot.json"

    print(f"Loading up to {NUM_DOCUMENTS} articles from {filtered_file}...")
    articles = load_articles(filtered_file, NUM_DOCUMENTS)
    print(f"Loaded {len(articles)} articles.")

    start = time.perf_counter()
    embeddings = embed_abstracts(articles, model_name=MODEL_NAME, batch_size=BATCH_SIZE)
    elapsed = time.perf_counter() - start

    print(f"\nEmbedding shape: {embeddings.shape}")
    print(f"Model          : {MODEL_NAME}")
    print(f"Batch size     : {BATCH_SIZE}")
    print(f"SIE URL        : {SIE_URL}")
    print(f"Time taken     : {elapsed:.2f}s")


if __name__ == "__main__":
    main()
