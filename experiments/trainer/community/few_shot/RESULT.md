# FewShotTrainer Result

## Summary

### Trainset Scores

| Benchmarks \ Methods                   | Baseline | finetuned baseline                        | FewShotTrainer                            |
| -------------------------------------- | -------- | ----------------------------------------- | ----------------------------------------- |
| BIRD-bench (SQL)                       | 0.291    | <span style="color:blue">**0.449**</span> | <span style="color:blue">0.357</span>     |
| BoolQ (QA)                             | 0.906    | <span style="color:blue">**1.000**</span> | <span style="color:blue">0.947</span>     |
| GPQA (Reasoning)                       | 0.186    | <span style="color:red">0.184</span>      | <span style="color:red">0.105</span>      |
| MATH (Reasoning)                       | 0.626    | <span style="color:red">0.566</span>      | <span style="color:blue">**0.681**</span> |
| New York Times Topics (Classification) | 0.836    | <span style="color:blue">**0.914**</span> | <span style="color:red">0.830</span>      |

### Testset Scores

| Benchmarks \ Methods                   | Baseline | finetuned baseline                        | FewShotTrainer                            |
| -------------------------------------- | -------- | ----------------------------------------- | ----------------------------------------- |
| BIRD-bench (SQL)                       | 0.307    | <span style="color:blue">**0.473**</span> | <span style="color:red">0.083</span>      |
| BoolQ (QA)                             | 0.850    | <span style="color:blue">0.892</span>     | <span style="color:blue">**0.900**</span> |
| GPQA (Reasoning)                       | 0.146    | <span style="color:red">0.080</span>      | <span style="color:red">0.110</span>      |
| MATH (Reasoning)                       | 0.610    | <span style="color:red">0.426</span>      | <span style="color:blue">**0.670**</span> |
| New York Times Topics (Classification) | 0.794    | <span style="color:blue">**0.818**</span> | <span style="color:red">0.770</span>      |

## Benchmarks Results

FewShotTrainer performs well in MATH benchmark, but not so well in other benchmarks.

### BIRD-bench

![BIRD-bench](../../../../images/trainer/community/few_shot/bird_bench_result.png)

### BoolQ

![BoolQ](../../../../images/trainer/community/few_shot/boolq_result.png)

### GPQA

![GPQA](../../../../images/trainer/community/few_shot/gpqa_result.png)

### MATH

![MATH](../../../../images/trainer/community/few_shot/math_result.png)

### New York Times Topics

![New York Times Topics](../../../../images/trainer/community/few_shot/new_york_times_topics_result.png)

## Future Work
