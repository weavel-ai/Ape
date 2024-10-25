# Dspy-Mipro

**Dspy-Mipro** is an algorithm from paper [DSPy](https://arxiv.org/abs/2310.03714) and project [DSPy](https://github.com/stanfordnlp/DSPy), based on the MiproV2 optimizer, implemented as of 2024.10.21.

## Description

Mipro (Multiprompt Instruction Proposal Optimizer) optimizes prompts by suggesting multiple improved instructions and groups of few-shot examples simultaneously. It uses the hyperparameter optimization library [Optuna](https://optuna.org/) to select the best combination of suggested instructions and few-shot examples.

## Differences Between Implementation and Paper

### Single Prompt Optimization

While the original DSPy project and the MIPRO algorithm are designed for end-to-end LLM chain optimization, Ape's implementation focuses on single prompt optimization, in line with Ape's development philosophy.

### Improvements in Instruction Suggestion

In DSPy’s MIPRO implementation, one selected tip is used for every suggested instruction. However, in Ape’s implementation, tips are randomly selected for each suggested instruction to enhance variety.

## Unique Insights/Techniques of the Paper

### Few-shot Example Selection

The few-shot example selection algorithm is unique. It selects examples where the model successfully solves the task, referred to as `bootstrapped_fewshot`. This is achieved by adding intermediate steps, such as chain-of-thought reasoning or retrieval-augmented generation steps into few-shot examples.

### Hyperparameter Optimization Approach

Optuna is used to select the optimal combination of suggested instructions and few-shot examples. This allows the algorithm to consider not only the individual impact of each prompt improvement but also the interaction effects between different improvements.

Theoretically, this expands the search space from k\*N (where k is the number of types of improvements, and N is the number of suggestions for each type) to N^k. Despite this larger search space, the algorithm efficiently navigates it by Bayesian TPE sampling.

## Potential Limitations

### Limited Instruction Suggestion Space

Since instruction suggestions are based solely on the LLM and predefined tips, the diversity of suggestions is somewhat limited. While the paper mentions using previous suggestions in instruction proposals, this has not been implemented in the DSPy repository as of 2024.10.21. Additionally, MIPRO is not an iterative algorithm, meaning previous suggestion history must be managed outside the optimization algorithm.

### Few-shot Example Selection Algorithm

The few-shot example selection algorithm is innovative, but it may not align well with single-prompt optimization for state-of-the-art models. In the paper, bootstrapped few-shot examples are used to train a student model to mimic a teacher model's behavior. This setup may not be optimal for prompt optimization when the student and teacher models are the same (as is the case with state-of-the-art models).

### Limited number of improvement

This method only apply 2 types of improvement for each optimization, instruction improvement and few-shot example improvement. So it's performance is hard to be improved as training dataset size increase.

### Difficult to use iteratively

This method apply both instruction and few-shot example improvement at same time. However, usually instruction based improvement can be applied iteratively, but few-shot does not. It makes hard to use this method iteratively, to previously optimized prompt.

## Benchmark Performance

To see performance benchmarks, visit [here](../../../../../../../experiments/trainer/papers/dspy_mipro/RESULT.md).
