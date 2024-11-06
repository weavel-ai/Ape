# FewShotTrainer

**FewShotTrainer** is a few-shot optimization method inspired by [DSPy](https://github.com/stanfordnlp/DSPy) and [adalflow](https://github.com/SylphAI-Inc/AdalFlow).

## Description

FewShotTrainer combines concepts from DSPy's MIPRO algorithm and adalflow's few-shot optimization techniques.

In DSPy's MIPRO, few-shot examples are optimized by running the BootstrapFewshot optimization in parallel and selecting the best group of examples.

Adalflow introduces a different few-shot selection algorithm. For bootstrap few-shot examples (those with intermediate steps like chain-of-thought reasoning), it selects examples with the highest score difference between the student and teacher models. For raw examples, it selects difficult ones by weighting the student model's score.

FewShotTrainer integrates these two approaches by selecting examples using adalflow's algorithm and optimizing them with DSPy MIPRO's method.

## Motivation for developing this method

While DSPy's MIPRO method for generating few-shot candidates is strict and effective, its selection algorithm is not ideal for prompt optimization in state-of-the-art models, as it relies on training a student model based on a teacher model’s behavior.

Adalflow’s few-shot selection algorithm addresses this issue, but since adalflow's goal is to integrate text-gradient optimization into production and focus on end-to-end prompt chain optimization, we developed FewShotTrainer to blend both approaches for better few-shot prompt optimization.

## Key Differences

FewShotTrainer separates the few-shot candidate generation from DSPy MIPRO and adds a final selection process based on training dataset scores for each few-shot candidate. Additionally, we replace DSPy's BootstrapFewshot selection algorithm with adalflow's selection algorithm.

## Benchmark Performance

If you want to see the benchmark performance, please visit [Here](../../../../../../../experiments/trainer/community/few_shot/RESULT.md).
