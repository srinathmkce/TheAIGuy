from datasets import load_dataset
from huggingface_hub import snapshot_download

snapshot_download("Qwen/Qwen3.5-0.8B", local_dir="Qwen3.5-0.8B")
ds = load_dataset("HuggingFaceM4/ChartQA", split="test")
ds.save_to_disk("data/ChartQA_test")
print(ds)
