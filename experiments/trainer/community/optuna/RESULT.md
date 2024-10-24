# OptunaTrainer Result

## Summary

### Trainset Scores

| Benchmarks \ Methods                   | Baseline | finetuned baseline                        | OptunaTrainer                             |
| -------------------------------------- | -------- | ----------------------------------------- | ----------------------------------------- |
| BIRD-bench (SQL)                       | 0.291    | <span style="color:blue">0.449</span>     | <span style="color:blue">**0.490**</span> |
| BoolQ (QA)                             | 0.906    | <span style="color:blue">**1.000**</span> | <span style="color:blue">**1.000**</span> |
| GPQA (Reasoning)                       | 0.186    | <span style="color:red">0.184</span>      | <span style="color:blue">**0.280**</span> |
| MATH (Reasoning)                       | 0.626    | <span style="color:red">0.566</span>      | <span style="color:blue">**0.760**</span> |
| New York Times Topics (Classification) | 0.836    | <span style="color:blue">0.914</span>     | <span style="color:blue">**0.960**</span> |

### Testset Scores

| Benchmarks \ Methods                   | Baseline | finetuned baseline                        | OptunaTrainer                             |
| -------------------------------------- | -------- | ----------------------------------------- | ----------------------------------------- |
| BIRD-bench (SQL)                       | 0.307    | <span style="color:blue">**0.473**</span> | <span style="color:red">0.294</span>      |
| BoolQ (QA)                             | 0.850    | <span style="color:blue">**0.892**</span> | <span style="color:blue">0.860</span>     |
| GPQA (Reasoning)                       | 0.146    | <span style="color:red">0.080</span>      | <span style="color:red">0.120</span>      |
| MATH (Reasoning)                       | 0.610    | <span style="color:red">0.426</span>      | <span style="color:blue">**0.720**</span> |
| New York Times Topics (Classification) | 0.794    | <span style="color:blue">**0.818**</span> | <span style="color:red">0.710</span>      |

The OptunaTrainer shows higher performance improvement on the training dataset compared to other methods, including the finetuned baseline.
However, the performance improvement on the test set is not as clear.
For some datasets like BIRD-bench, GPQA, and New York Times Topics, the OptunaTrainer optimized prompt's performance is even lower than the baseline. This suggests that the OptunaTrainer is prone to overfitting.

## Benchmarks Results

### BIRD-bench

![BIRD-bench](../../../../images/trainer/community/optuna/bird_bench_result.png)

### BoolQ

![BoolQ](../../../../images/trainer/community/optuna/boolq_result.png)

### GPQA

![GPQA](../../../../images/trainer/community/optuna/gpqa_result.png)

### MATH

![MATH](../../../../images/trainer/community/optuna/math_result.png)

### New York Times Topics

![New York Times Topics](../../../../images/trainer/community/optuna/new_york_times_topics_result.png)

## Future Work
