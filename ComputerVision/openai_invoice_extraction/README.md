## Introduction
This example demonstrates how to leverage MLflow for experimenting OpenAI’s GPT-5-mini and GPT-5-nano models on the invoice extraction task, using the CORD-v2 dataset.

The goal is not only to compare model performance, but also to showcase how MLflow can streamline the entire workflow with:

* Prompt Versioning – Track different prompt templates used for extraction.

* Tracing & Logging – Capture inputs, outputs, and intermediate steps for reproducibility.

* Evaluation Metrics – Record accuracy, consistency, and cost trade-offs for each experiment.

### Dataset

We use the CORD-v2 dataset available on Hugging Face: [CORD-v2 on HuggingFace](https://huggingface.co/datasets/naver-clova-ix/cord-v2).

For this experiment, we focus exclusively on the test split, which contains 100 invoice samples. This subset provides a standardized experiments for evaluating invoice field extraction using GPT-5-mini and GPT-5-nano, while keeping the experiment lightweight and reproducible.

The dataset is under [Creative Commons Attribution 4.0 International
License][cc-by].

[![CC BY 4.0][cc-by-image]][cc-by]

[cc-by]: http://creativecommons.org/licenses/by/4.0/
[cc-by-image]: https://i.creativecommons.org/l/by/4.0/88x31.png
[cc-by-shield]: https://img.shields.io/badge/License-CC%20BY%204.0-lightgrey.svg


### Prerequisites

#### Start the MLflow server

Run the following command to start the MLflow server
```
mlflow server --host 127.0.0.1 --port 8080
```

#### Setting OpenAI environment

Create a new file `.env` and paste your OPENI key

```
OPENAI_API_KEY=<OPENAI-API-KEY>
```

### Running the experiment

Open the [cord_v2_gpt5_baseline_mlflow.ipynb](./cord_v2_gpt5_baseline_mlflow.ipynb) notebook.

Install the necessary dependencies mentioned in the notebook.

The provided Jupyter notebook walks through the full benchmarking workflow step by step. It is organized into the following sections:  


1. **Set the configrations**
   - Set the MLflow, OpenAI and prompt configurations here.
   ![Setting up MLflow](./screenshots/mlflow.png)

2. **Register prompt and model**
   - Register the system prompt in MLflow
   ![Register the prompt](./screenshots/prompt.png)

3. **Load the Dataset**  
   - Downloads and loads the **CORD-v2** dataset from Hugging Face.  
   ![Load the dataset](./screenshots/dataset.png)

4. **Data Preparation**  
   - Selects the number of samples to evaluate.  
   - Prepares the invoice data in a model-ready format for inference.

5. **Deriving individual invoice level metrics**  
   - Calculate the accuracy of the individual invoice
   ![Calculate Accuracy](./screenshots/invoice-metrics.png)

6. **Tracing OpenAI API calls**  
   - Tracing individual invoice extraction parameters - request, response, metadata
   ![Trace analysis](./screenshots/trace-analysis.png)

5. **Logging model level metrics**  
   - Logging aggregated metrics for comparison
   ![Logging model level metrics](./screenshots/model-metrics.png)


Overall model comparison can be done at the experiment level
![Overall model comparison](./screenshots/overall-model-comparison.png)

Overall trace comparison can be done by clicking the "Traces" Tab
![Overall trace comparison](./screenshots/overall-trace-comparison.png)

Overall model metrics
![Accuracy](./screenshots/accuracy.png)
![Other Metrics](./screenshots/other-metrics.png)

Best performing models
![Best performing models](./screenshots/best-models.png)

