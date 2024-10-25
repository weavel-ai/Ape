# TextGradEvoTrainer

**TextGradEvoTrainer** is inspired by the papers [TextGrad](https://arxiv.org/abs/2406.07496) and [EvoPrompt](https://arxiv.org/abs/2309.08532), combining evolutionary algorithms with text gradient-based optimization.

## Description

TextGradEvoTrainer is a method that combines the strengths of both TextGrad and EvoPrompt to optimize prompts. It leverages the learning capabilities of TextGrad from the training dataset and the search efficiency of EvoPrompt’s evolutionary algorithm.

## Motivation for developing this method

While TextGrad learns significantly from the training dataset by extracting textual gradients, the success rate of those gradients in improving performance is low—less than 10%. On the other hand, EvoPrompt uses an evolutionary algorithm to effectively search the prompt space, but it does not learn directly from the training dataset, limiting peak performance improvement.

TextGradEvoTrainer integrates these two methods. It first extracts textual gradients from the training dataset and then uses an evolutionary algorithm to optimize prompts, with the gradient guiding the evolutionary process.

## Key Differences

### Running on Each Example

TextGradientTrainer operates on batches of examples, while TextGradEvoTrainer runs on individual examples.  
To use the evolutionary algorithm, a population of candidate prompts needs to be created. If run on batches, the number of textual gradients can vary across batches, requiring different population generation algorithms for each batch. To avoid this complexity, TextGradEvoTrainer processes one example at a time, generating a population by extracting various textual gradients from each example concurrently.

### Validation Step with Evolutionary Algorithm

In TextGrad, the validation step checks whether the optimized prompt is truly improved across the entire training dataset. Since textual gradients are much more unstable compared to tensor gradients, the original TextGrad simply attempts multiple iterations (N times) to extract and apply textual gradients until it succeeds.

TextGradEvoTrainer improves on this by using an evolutionary algorithm during the validation step. It first generates a population of candidate prompts and evaluates them against the training dataset. Then, the evolutionary algorithm is applied to this population, refining the prompts until one successfully validates (i.e., improves performance across the training dataset).

## Benchmark Performance

For benchmark performance, visit [link](../../../../../../../experiments/trainer/community/text_grad_evo/RESULT.md).
