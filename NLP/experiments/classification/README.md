## Introduction

This project involves experimenting with OpenAI, open-source, and various machine learning models to classify Wikipedia articles effectively. The aim is to explore different methodologies and enhance the accuracy and efficiency of text classification tasks.

## Dataset

Wikipedia Dataset from Huggingface is used for the test. Download the dataset from here - https://huggingface.co/datasets/wikimedia/wikipedia

Only the articles tagged to english are considered for this experiment - https://huggingface.co/datasets/wikimedia/wikipedia/viewer/20231101.en - 6.5 Million rows

## Experiment Results

| Experiment   | Model        | Total Processed | Time Taken | Actual Cost(in dollars) | Estimated Cost(in dollars) | Estimated Time | Remarks            |
|--------------|--------------|-----------------|------------|-------------|----------------|----------------|--------------------|
| [Experiment-1](../classification/Experiment-1%20GPT3.5.ipynb) | [GPT 3.5 Turbo](https://platform.openai.com/docs/models/gpt-3-5-turbo) | 5000 articles   | 50 mins    | $5         | $3000            | 40 days      | Sequential execution - Naive approach and not effective for huge dataset   |
