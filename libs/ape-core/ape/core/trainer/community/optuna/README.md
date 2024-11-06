# OptunaTrainer

(Note: The name of this trainer will be updated soon.)  
OptunaTrainer is inspired by [DSPy](https://github.com/stanfordnlp/DSPy) but focuses more on instruction optimization.

## Description

OptunaTrainer is a variation of DSPy-MIPRO. Unlike DSPy-MIPRO, which improves both instructions and few-shot examples, OptunaTrainer focuses solely on two types of instruction optimization. For the second type of instruction optimization, it uses metadata from evaluation metrics to guide the optimization process.

## Motivation for developing this method

While DSPy-MIPRO’s hyperparameter optimization approach is powerful, it has some limitations. It doesn't learn directly from the training dataset, and its instruction optimization relies entirely on predefined tips.

Team Weavel discovered that, in some real-world cases, people prefer to evaluate and optimize prompts based on complex metrics, not just accuracy. For example, some users may want to optimize for metrics like ROUGE, BLEU, Precision, Recall, or MICRO-F1. To address these needs, OptunaTrainer was developed.

## Key Differences

Compared to DSPy-MIPRO, OptunaTrainer generates additional optimized instruction candidates instead of few-shot example candidates. For these new candidates, it extracts insights from the metadata of evaluation metrics and combines this information with predefined tips. Each candidate is generated using different, randomly selected tips. After generating the two types of candidates, OptunaTrainer optimizes them using the Optuna TPESampler.

## Benchmark Performance

To view performance benchmarks, visit [here](../../../../../../../experiments/trainer/community/optuna/RESULT.md).
