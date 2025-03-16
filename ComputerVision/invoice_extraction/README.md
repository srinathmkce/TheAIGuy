# 🚀 Invoice Extraction Using Vision Language Models  

This repository contains a 9-part tutorial series on building an Invoice Extraction system using Vision Language Models (VLM). The series covers everything from data exploration to model fine-tuning and deployment.  

Watch the series here - https://youtu.be/ijoL0J9wqVM

---

## 📌 **Series Overview**  

### ✅ **Part 1: Introduction**  
- Overview of Vision Language Models (VLM)  
- Why VLMs are effective for invoice extraction  
- Use cases and benefits  

### ✅ **Part 2: Exploring the Invoice Dataset**  
- Introduction to the CORD dataset - https://huggingface.co/datasets/naver-clova-ix/cord-v2
- Data cleaning and preprocessing steps  
- Identifying key fields in the invoice  

### ✅ **Part 3: Creating a Baseline**  
- Setting up a simple baseline model - [Notebook](./cord_v2_baseline_qwen_7b.ipynb) 
- Running inference on sample data  
- Measuring initial performance  

### ✅ **Part 4: Deriving Accuracy Metric**  
- Understanding accuracy for invoice extraction - [Notebook](./cord_v2_baseline_qwen_7b.ipynb)  
- Calculating token-level and field-level accuracy  

### ✅ **Part 5: Deriving Precision and Recall**  
- Computing precision, recall, and F1 score  - [Notebook](./cord_v2_baseline_qwen_7b.ipynb)
- Handling multi-label extraction scenarios  
- Interpreting the results  

### ✅ **Part 6: Optimizing Inference with VLLM**  
- Setting up VLLM for faster inference  - [Notebook](./vllm_inference_and_webui.ipynb)
- Batch processing and memory optimization  
- Reducing latency and improving throughput  

### ✅ **Part 7: Fine-Tuning Vision Language Models**  
- Using Unsloth for efficient fine-tuning - [Notebook](./cord_v2_qwen_7b_finetuning.ipynb)  
- Handling large-scale datasets  
- Performance improvements post-fine-tuning  

### ✅ **Part 8: Creating a Web Interface**  
- Building a Flask-based web app - [Notebook](./vllm_inference_and_webui.ipynb)  
- Displaying extracted fields in real-time  
- User-friendly interface design  

### ✅ **Part 9: Recap and Conclusions**  
- Summary of key takeaways  
- Performance benchmarks and future improvements  

## References

1. Dataset - https://github.com/clovaai/cord
2. Finetuning - https://github.com/unslothai/unsloth
3. Maestro Prompt - https://github.com/roboflow/maestro 
4. Optimizing inference using VLLM - https://github.com/vllm-project/vllm

## Citation

### CORD: A Consolidated Receipt Dataset for Post-OCR Parsing
```
@article{park2019cord,
  title={CORD: A Consolidated Receipt Dataset for Post-OCR Parsing},
  author={Park, Seunghyun and Shin, Seung and Lee, Bado and Lee, Junyeop and Surh, Jaeheung and Seo, Minjoon and Lee, Hwalsuk}
  booktitle={Document Intelligence Workshop at Neural Information Processing Systems}
  year={2019}
}
```
### Post-OCR parsing: building simple and robust parser via BIO tagging

```
@article{hwang2019post,
  title={Post-OCR parsing: building simple and robust parser via BIO tagging},
  author={Hwang, Wonseok and Kim, Seonghyeon and Yim, Jinyeong and Seo, Minjoon and Park, Seunghyun and Park, Sungrae and Lee, Junyeop and Lee, Bado and Lee, Hwalsuk}
  booktitle={Document Intelligence Workshop at Neural Information Processing Systems}
  year={2019}
}
``` 
### OCR-free Document Understanding Transformer 🍩

```
@article{kim2021donut,
   title={OCR-free Document Understanding Transformer},
   author={Kim, Geewook and Hong, Teakgyu and Yim, Moonbin and Nam, JeongYeon and Park, Jinyoung and Yim, Jinyeong and Hwang, Wonseok and Yun, Sangdoo and Han, Dongyoon and Park, Seunghyun},
   journal={arXiv preprint arXiv:2111.15664},
   year={2021}
}
```