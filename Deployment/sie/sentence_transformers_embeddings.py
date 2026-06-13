import argparse
import json
import time
import numpy as np
import kagglehub
from sentence_transformers import SentenceTransformer
from tqdm import tqdm

MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
NUM_DOCUMENTS = 100000
BATCH_SIZE = 128


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
    device: str = "cpu",
) -> np.ndarray:
    model = SentenceTransformer(model_name, device=device)

    abstracts = [article.get("abstract", "") for article in articles]

    batches = [abstracts[i : i + batch_size] for i in range(0, len(abstracts), batch_size)]

    all_embeddings = []
    for batch in tqdm(batches, desc="Embedding abstracts", unit="batch"):
        embeddings = model.encode(batch, show_progress_bar=False)
        all_embeddings.append(embeddings)

    return np.vstack(all_embeddings)


def main():
    parser = argparse.ArgumentParser(description="Embed arxiv abstracts using SentenceTransformers")
    parser.add_argument(
        "--device",
        choices=["cpu", "cuda"],
        default="cpu",
        help="Device to run the model on (default: cpu)",
    )
    args = parser.parse_args()

    path = kagglehub.dataset_download("Cornell-University/arxiv")
    filtered_file = path + "/arxiv-metadata-oai-snapshot.json"

    print(f"Loading up to {NUM_DOCUMENTS} articles from {filtered_file}...")
    articles = load_articles(filtered_file, NUM_DOCUMENTS)
    print(f"Loaded {len(articles)} articles.")

    start = time.perf_counter()
    embeddings = embed_abstracts(articles, model_name=MODEL_NAME, batch_size=BATCH_SIZE, device=args.device)
    elapsed = time.perf_counter() - start

    print(f"\nEmbedding shape: {embeddings.shape}")
    print(f"Model          : {MODEL_NAME}")
    print(f"Batch size     : {BATCH_SIZE}")
    print(f"Device         : {args.device}")
    print(f"Time taken     : {elapsed:.2f}s")


if __name__ == "__main__":
    main()
