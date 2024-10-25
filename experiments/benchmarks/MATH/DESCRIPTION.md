# MATH

## Source

[hendrycks/competition_math](https://huggingface.co/datasets/hendrycks/competition_math)  
[Paper](https://arxiv.org/abs/2103.03874)  
[GitHub](https://github.com/hendrycks/math)

## Description

**MATH** is a dataset designed for evaluating mathematical problem-solving abilities. It contains 7,500 training samples and 5,000 test samples, covering a wide range of mathematical topics from arithmetic to geometry.

## Motivation

MATH is an essential dataset for evaluating the reasoning and mathematical capabilities of LLMs. As of 2024.10.21, performance on this dataset is not fully saturated, except by the OpenAI O1 model, which is not publicly available. Given the uncertainty about whether prompt optimization can improve reasoning skills, MATH serves as a critical benchmark to assess the potential of prompt optimization in enhancing LLMs' performance on reasoning tasks.

## Preprocessing

1. Shuffle the `train` and `test` splits using seed `1`.
2. sample first 100 samples from the `train` split for the training set and first 100 samples from the `test` split for the validation set.
3. Use the `problem` column as the input, and format the `solution` column as the output in the following structure: `{"thought": "...", "answer": "..."}`. Separate the thought process and the final answer using the pattern `\\boxed{}` to split the content.

## Evaluation

The LLM evaluation is conducted using the following prompt:

```markdown
YOU ARE one of the GREATEST mathematicians, logicians, programmers, and AI scientists. You are intelligent and rational. You are prudent and cautious. Your mastery over Arithmetic, Combinatorics, Number Theory, Probability Theory, Algebra, Analysis, and Geometry is unparalleled. You THINK NATURAL, BROAD AND DEEP. Let's think step by step.
Your job is to judge whether the "final_answer" is correct based on "ground_truth_answer", do not be strict on the format, but check the content. Notice that unsolved half results are not Correct.
Problem: {question_content}
Is the final_answer correct, given the ground truth answer? Reply with Correct, Wrong or Unknown.
"final_answer": "{final_answer}", "ground_truth_answer": "{ground_truth_answer}"

You MUST respond in JSON format like below:
{{
    "thought": "...",
    "correctness": "<correctness>", One of "Correct", "Wrong", "Unknown"
}}
```
