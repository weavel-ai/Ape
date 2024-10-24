# TextGradientTrainer

**TextGradientTrainer** is an implementation of the [TextGrad](https://arxiv.org/abs/2406.07496) paper, based on the [TextGrad](https://github.com/zou-group/textgrad) repository and the [adalflow](https://github.com/SylphAI-Inc/AdalFlow) project, as of 2024.10.21.  
 Although inspired by the original paper, this implementation has diverged significantly, so it is included in the community section.

## Description

TextGradientTrainer is a method that iteratively optimizes prompts by generating "text gradients." Drawing inspiration from the Gradient Descent, Backpropagation, and Autograd frameworks, it replaces tensor gradients with text gradients.

The process is similar to machine learning (ML) training. For each batch, a loss score is generated based on a specified metric, and a gradient is generated through textual data. An optimized prompt is then suggested by applying the gradient. While gradient descent is proven to improve batch data performance, text gradients are less established, so TextGradientTrainer validates the suggested prompt against both batch data and the training dataset.

While the original TextGrad was designed for end-to-end LLM chain optimization, TextGradientTrainer focuses on single prompt optimization, following Ape's development philosophy.

## Motivation for Developing this Method

TextGrad is a powerful concept because it learns significantly from the training dataset compared to other methods. This makes it highly suitable for single prompt optimization, which prompted the creation of TextGradientTrainer.

## Key Differences

### Text Gradient Generation Process

TextGrad, inspired by gradient descent, follows a three-step process to generate text gradients. First, it generates textual loss from the chosen metric. Second, it produces a gradient using the loss, prompt, input data, and output predictions. Finally, it applies the gradient to suggest an optimized prompt.

However, we found that this process doesn't fully utilize available information when generating gradients. Therefore, TextGradientTrainer employs a streamlined two-step process: first, it generates textual gradients by incorporating all available information (including metric scores, input data, predictions, and ground truth), and then it applies the gradient to optimize the prompt.

### Suggested Prompt Validation Process

We discovered that the sampling-based validation used in the original TextGrad is unstable. To ensure better stability, TextGradientTrainer validates suggested prompts against the entire training dataset. Although this approach is slower and more resource-intensive, it provides more consistent and reliable optimization results.

## Benchmark Performance

For performance benchmarks, visit [link](../../../../../../../experiments/trainer/community/text_gradient/RESULT.md).
