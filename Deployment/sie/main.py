from sie_embeddings import load_articles, embed_abstracts, MODEL_NAME, BATCH_SIZE, SIE_URL, NUM_DOCUMENTS
import time

def main():
    file_path = "/root/.cache/kagglehub/datasets/Cornell-University/arxiv/versions/288/arxiv-metadata-oai-snapshot.json"
    
    print(f"Loading up to {NUM_DOCUMENTS} articles from {file_path}...")
    articles = load_articles(file_path, NUM_DOCUMENTS)
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
