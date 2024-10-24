# Benchmark Results

Currently (since 2024.10.21), All benchmark results are based on the `gpt-4o-mini` model with temperature 0.0.
For finetuned baseline, we use openai finetuning api.

## Summary

All methods tend to show good performance on the training set, but their performance on the test set is not always consistent. This suggests that prompt optimization methods have a tendency to overfit.
These methods demonstrate notable performance improvements in reasoning benchmarks like MATH and GPQA compared to the finetuned baseline, but they don't perform as well in other benchmarks.

## Trainset Results

all results are best score in trainset.

| Benchmarks \ Methods                   | Baseline | finetuned baseline                        | DSPy-MIPRO                                | EvoPrompt                             | TextGradientTrainer                   | FewShotTrainer                        | ExpelTrainer | OptunaTrainer                             | TextGradEvoTrainer |
| -------------------------------------- | -------- | ----------------------------------------- | ----------------------------------------- | ------------------------------------- | ------------------------------------- | ------------------------------------- | ------------ | ----------------------------------------- | ------------------ |
| BIRD-bench (SQL)                       | 0.291    | <span style="color:blue">0.449</span>     | <span style="color:blue">0.439</span>     | <span style="color:blue">0.368</span> | <span style="color:blue">0.394</span> | <span style="color:blue">0.357</span> | -            | <span style="color:blue">**0.490**</span> |                    |
| BoolQ (QA)                             | 0.906    | <span style="color:blue">**1.000**</span> | <span style="color:blue">0.960</span>     | <span style="color:red">0.900</span>  | <span style="color:blue">0.910</span> | <span style="color:blue">0.947</span> | -            | <span style="color:blue">**1.000**</span> |                    |
| GPQA (Reasoning)                       | 0.186    | <span style="color:red">0.184</span>      | <span style="color:blue">0.240</span>     | <span style="color:blue">0.190</span> | <span style="color:blue">0.230</span> | <span style="color:red">0.105</span>  | -            | <span style="color:blue">**0.280**</span> |                    |
| MATH (Reasoning)                       | 0.626    | <span style="color:red">0.566</span>      | <span style="color:blue">**0.760**</span> | <span style="color:blue">0.680</span> | <span style="color:blue">0.730</span> | <span style="color:blue">0.681</span> | -            | <span style="color:blue">**0.760**</span> |                    |
| New York Times Topics (Classification) | 0.836    | <span style="color:blue">0.914</span>     | <span style="color:blue">0.920</span>     | <span style="color:blue">0.840</span> | <span style="color:blue">0.860</span> | <span style="color:red">0.830</span>  | -            | <span style="color:blue">**0.960**</span> |                    |

## Testset Results

All results show the testset performance of the prompt that performed best on the trainset.

| Benchmarks \ Methods                   | Baseline | finetuned baseline                        | DSPy-MIPRO                                | EvoPrompt                             | TextGradientTrainer                       | FewShotTrainer                        | ExpelTrainer | OptunaTrainer                             | TextGradEvoTrainer |
| -------------------------------------- | -------- | ----------------------------------------- | ----------------------------------------- | ------------------------------------- | ----------------------------------------- | ------------------------------------- | ------------ | ----------------------------------------- | ------------------ |
| BIRD-bench (SQL)                       | 0.307    | <span style="color:blue">**0.473**</span> | <span style="color:red">0.242</span>      | <span style="color:red">0.292</span>  | <span style="color:red">0.285</span>      | <span style="color:red">0.083</span>  | -            | <span style="color:red">0.294</span>      |                    |
| BoolQ (QA)                             | 0.850    | <span style="color:blue">0.892</span>     | <span style="color:blue">0.860</span>     | <span style="color:blue">0.870</span> | <span style="color:blue">**0.920**</span> | <span style="color:blue">0.900</span> | -            | <span style="color:blue">0.860</span>     |                    |
| GPQA (Reasoning)                       | 0.146    | <span style="color:red">0.080</span>      | <span style="color:blue">**0.180**</span> | <span style="color:red">0.120</span>  | <span style="color:red">0.140</span>      | <span style="color:red">0.110</span>  | -            | <span style="color:red">0.120</span>      |                    |
| MATH (Reasoning)                       | 0.610    | <span style="color:red">0.426</span>      | <span style="color:blue">0.650</span>     | <span style="color:blue">0.670</span> | <span style="color:blue">**0.720**</span> | <span style="color:blue">0.670</span> | -            | <span style="color:blue">**0.720**</span> |                    |
| New York Times Topics (Classification) | 0.794    | <span style="color:blue">**0.818**</span> | <span style="color:red">0.700</span>      | <span style="color:red">0.600</span>  | <span style="color:red">0.730</span>      | <span style="color:red">0.770</span>  | -            | <span style="color:red">0.710</span>      |                    |
