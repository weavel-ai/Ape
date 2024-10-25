# BoolQ

## Source

[google/boolq](https://huggingface.co/datasets/google/boolq)

## Description

**BoolQ** is a question-answering dataset for yes/no questions, created by Google. It contains 9,427 training samples and 3,270 validation samples.

## Motivation

We aim to evaluate prompt optimization performance on simple QA tasks without requiring sophisticated reasoning. BoolQ is an ideal candidate because it doesn't fully saturate the performance of large language models (LLMs). In our tests with **gpt-4o-mini**, the model achieved a performance of approximately 87.5%.

If we discover a simple QA task that is more challenging than this dataset, we will stop using BoolQ as a benchmark for QA tasks in the Ape project.

## Preprocessing

1. Shuffle the dataset using seed `1`.
2. Select the first 100 samples (`[:100]`) from the training split as the training set, and the first 100 samples (`[:100]`) from the validation split as the validation set.
3. Use the `question` and `passage` columns as inputs, and format the `answer` column as the output in the following structure: `{"answer": "..."}`.

## Evaluation

The evaluation is based on an exact match between the predicted answer and the ground truth answer.
