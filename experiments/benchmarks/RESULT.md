# Benchmark Results

Currently (since 2024.10.21), All benchmark results are based on the `gpt-4o-mini` model with temperature 0.0.
For finetuned baseline, we use openai finetuning api.

## Summary

All methods tend to show good performance on the training set, but their performance on the test set is not always consistent. This suggests that prompt optimization methods have a tendency to overfit.
These methods demonstrate notable performance improvements in reasoning benchmarks like MATH and GPQA compared to the finetuned baseline, but they don't perform as well in other benchmarks.

## Trainset Results

all results are best score in trainset.

| Benchmarks \ Methods                   | Baseline | finetuned baseline | DSPy-MIPRO    | EvoPrompt | TextGradientTrainer | FewShotTrainer | ExpelTrainer | OptunaTrainer | TextGradEvoTrainer |
| -------------------------------------- | -------- | ------------------ | ------------- | --------- | ------------------- | -------------- | ------------ | ------------- | ------------------ |
| BIRD-bench (SQL)                       | 0.291    | 0.449 (▲)          | 0.439 (▲)     | 0.368 (▲) | 0.394 (▲)           | 0.357 (▲)      | -            | **0.490** (▲) |                    |
| BoolQ (QA)                             | 0.906    | **1.000** (▲)      | 0.960 (▲)     | 0.900 (▼) | 0.910 (▲)           | 0.947 (▲)      | -            | **1.000** (▲) |                    |
| GPQA (Reasoning)                       | 0.186    | 0.184 (▼)          | 0.240 (▲)     | 0.190 (▲) | 0.230 (▲)           | 0.105 (▼)      | -            | **0.280** (▲) |                    |
| MATH (Reasoning)                       | 0.626    | 0.566 (▼)          | **0.760** (▲) | 0.680 (▲) | 0.730 (▲)           | 0.681 (▲)      | -            | **0.760** (▲) |                    |
| New York Times Topics (Classification) | 0.836    | 0.914 (▲)          | 0.920 (▲)     | 0.840 (▲) | 0.860 (▲)           | 0.830 (▼)      | -            | **0.960** (▲) |                    |

## Testset Results

All results show the testset performance of the prompt that performed best on the trainset.

| Benchmarks \ Methods                   | Baseline | finetuned baseline | DSPy-MIPRO    | EvoPrompt | TextGradientTrainer | FewShotTrainer | ExpelTrainer | OptunaTrainer | TextGradEvoTrainer |
| -------------------------------------- | -------- | ------------------ | ------------- | --------- | ------------------- | -------------- | ------------ | ------------- | ------------------ |
| BIRD-bench (SQL)                       | 0.307    | **0.473** (▲)      | 0.242 (▼)     | 0.292 (▼) | 0.285 (▼)           | 0.083 (▼)      | -            | 0.294 (▼)     |                    |
| BoolQ (QA)                             | 0.850    | 0.892 (▲)          | 0.860 (▲)     | 0.870 (▲) | **0.920** (▲)       | 0.900 (▲)      | -            | 0.860 (▲)     |                    |
| GPQA (Reasoning)                       | 0.146    | 0.080 (▼)          | **0.180** (▲) | 0.120 (▼) | 0.140 (▼)           | 0.110 (▼)      | -            | 0.120 (▼)     |                    |
| MATH (Reasoning)                       | 0.610    | 0.426 (▼)          | 0.650 (▲)     | 0.670 (▲) | **0.720** (▲)       | 0.670 (▲)      | -            | **0.720** (▲) |                    |
| New York Times Topics (Classification) | 0.794    | **0.818** (▲)      | 0.700 (▼)     | 0.600 (▼) | 0.730 (▼)           | 0.770 (▼)      | -            | 0.710 (▼)     |                    |
