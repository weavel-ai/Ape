# EvoPrompt

**EvoPrompt** is an algorithm described in the paper [EvoPrompt](https://arxiv.org/abs/2309.08532) and its accompanying GitHub repository [EvoPrompt](https://github.com/beeevita/EvoPrompt), implemented as of 2024.10.21.

## Description

EvoPrompt is a prompt optimization method based on evolutionary algorithms. In each generation, the prompt evolves in one of three ways: paraphrasing, genetic algorithms, or differential evolution. For genetic algorithms and differential evolution, the improved portions of each prompt are extracted and used in the evolution process.

## Differences Between Implementation and Paper

The original implementation in the paper is based on the Text Completion API, whereas Ape’s implementation is adapted for the Chat Completion API.

## Unique Insights/Techniques from the Paper

The paper highlights that evolutionary algorithm-based paraphrasing is significantly more effective than simple random paraphrasing. This insight can be applied to other prompt optimization methods by introducing additional paraphrasing steps for each improvement.

## Potential Limitations

### Limited Suggestion Space

Since the next generation of prompts is generated only through the LLM's paraphrasing, it doesn't directly learn from the training dataset, leading to limited diversity. Due to this limitation, the average performance within each generation gradually improves with each generation, but the peak performance doesn't show significant improvement.

## Benchmark Performance

If you want to see the benchmark performance, please visit [Here](../../../../../../../experiments/trainer/papers/evo_prompt/RESULT.md).
