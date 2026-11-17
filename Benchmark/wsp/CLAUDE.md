# CLAUDE.md

## Objective
The overall project goal is to achieve the improved benchmark using the weight space programming technique.

## Dataset
Dataset used for this - HuggingFaceM4/ChartQA - only the test dataset

## Environment
Always use the virtual environment to install packages. 
Download the dataset and the model in the current folder to load it locally

## Process
Two step process:
1. Setup a benchmark for ChartQA using Qwen/Qwen3-0.6B model. Create a validation code which runs the benchmark. Load, validate, publish the result, offload.
2. Create an agent which changes the weights of the qwen model and invokes step 1 to find the accuracy. Weight change, benchmark, feedback and loop.

Agent can never change the validation or benchmark.