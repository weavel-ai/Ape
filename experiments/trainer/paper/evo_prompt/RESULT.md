# EvoPrompt Result

## Summary

### Trainset Scores

| Benchmarks \ Methods                   | Baseline | finetuned baseline                        | EvoPrompt                                 |
| -------------------------------------- | -------- | ----------------------------------------- | ----------------------------------------- |
| BIRD-bench (SQL)                       | 0.291    | <span style="color:blue">**0.449**</span> | <span style="color:blue">0.368</span>     |
| BoolQ (QA)                             | 0.906    | <span style="color:blue">**1.000**</span> | <span style="color:red">0.900</span>      |
| GPQA (Reasoning)                       | 0.186    | <span style="color:red">0.184</span>      | <span style="color:blue">**0.190**</span> |
| MATH (Reasoning)                       | 0.626    | <span style="color:red">0.566</span>      | <span style="color:blue">**0.680**</span> |
| New York Times Topics (Classification) | 0.836    | <span style="color:blue">**0.914**</span> | <span style="color:blue">0.840</span>     |

### Testset Scores

| Benchmarks \ Methods                   | Baseline | finetuned baseline                        | EvoPrompt                                 |
| -------------------------------------- | -------- | ----------------------------------------- | ----------------------------------------- |
| BIRD-bench (SQL)                       | 0.307    | <span style="color:blue">**0.473**</span> | <span style="color:red">0.292</span>      |
| BoolQ (QA)                             | 0.850    | <span style="color:blue">**0.892**</span> | <span style="color:blue">0.870</span>     |
| GPQA (Reasoning)                       | 0.146    | <span style="color:red">0.080</span>      | <span style="color:red">0.120</span>      |
| MATH (Reasoning)                       | 0.610    | <span style="color:red">0.426</span>      | <span style="color:blue">**0.670**</span> |
| New York Times Topics (Classification) | 0.794    | <span style="color:blue">**0.818**</span> | <span style="color:red">0.600</span>      |

The frequency of performance improvements throughout the training process is notably low.
For BoolQ, no performance improvements were observed, while BIRD-bench and MATH showed only one instance of improvement each.

Furthermore, the performance improvements observed in the training dataset do not correlate well with those in the test dataset.

## Benchmarks Results

### BIRD-bench

![BIRD-bench](../../../../images/trainer/paper/evo_prompt/bird_bench_result.png)

### BoolQ

![BoolQ](../../../../images/trainer/paper/evo_prompt/boolq_result.png)

### GPQA

![GPQA](../../../../images/trainer/paper/evo_prompt/gpqa_result.png)

### MATH

![MATH](../../../../images/trainer/paper/evo_prompt/math_result.png)

### New York Times Topics

![New York Times Topics](../../../../images/trainer/paper/evo_prompt/new_york_times_topics_result.png)

## Future Work
