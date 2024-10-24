# DSPy-MIPRO Result

## Summary

### Trainset Scores

| Benchmarks \ Methods                   | Baseline | finetuned baseline                        | DSPy-MIPRO                                |
| -------------------------------------- | -------- | ----------------------------------------- | ----------------------------------------- |
| BIRD-bench (SQL)                       | 0.291    | <span style="color:blue">**0.449**</span> | <span style="color:blue">0.439</span>     |
| BoolQ (QA)                             | 0.906    | <span style="color:blue">**1.000**</span> | <span style="color:blue">0.960</span>     |
| GPQA (Reasoning)                       | 0.186    | <span style="color:red">0.184</span>      | <span style="color:blue">**0.240**</span> |
| MATH (Reasoning)                       | 0.626    | <span style="color:red">0.566</span>      | <span style="color:blue">**0.760**</span> |
| New York Times Topics (Classification) | 0.836    | <span style="color:blue">0.914</span>     | <span style="color:blue">**0.920**</span> |

### Testset Scores

| Benchmarks \ Methods                   | Baseline | finetuned baseline                        | DSPy-MIPRO                                |
| -------------------------------------- | -------- | ----------------------------------------- | ----------------------------------------- |
| BIRD-bench (SQL)                       | 0.307    | <span style="color:blue">**0.473**</span> | <span style="color:red">0.242</span>      |
| BoolQ (QA)                             | 0.850    | <span style="color:blue">**0.892**</span> | <span style="color:blue">0.860</span>     |
| GPQA (Reasoning)                       | 0.146    | <span style="color:red">0.080</span>      | <span style="color:blue">**0.180**</span> |
| MATH (Reasoning)                       | 0.610    | <span style="color:red">0.426</span>      | <span style="color:blue">**0.650**</span> |
| New York Times Topics (Classification) | 0.794    | <span style="color:blue">**0.818**</span> | <span style="color:red">0.700</span>      |

## Benchmarks Results

DSPy-MIPRO shows good performance in reasoning benchmarks like MATH and GPQA.

### BIRD-bench

![BIRD-bench](../../../../images/trainer/paper/dspy_mipro/bird_bench_result.png)

### BoolQ

![BoolQ](../../../../images/trainer/paper/dspy_mipro/boolq_result.png)

### GPQA

![GPQA](../../../../images/trainer/paper/dspy_mipro/gpqa_result.png)

### MATH

![MATH](../../../../images/trainer/paper/dspy_mipro/math_result.png)

### New York Times Topics

![New York Times Topics](../../../../images/trainer/paper/dspy_mipro/new_york_times_topics_result.png)

## Future Work
