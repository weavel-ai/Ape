# GPQA

## Source

[Idavidrein/gpqa](https://huggingface.co/datasets/Idavidrein/gpqa)  
[GitHub](https://github.com/idavidrein/gpqa?tab=readme-ov-file)  
[Paper](https://arxiv.org/abs/2311.12022)

## Description

**GPQA** is a graduate-level question-answering dataset focused on biology, physics, and chemistry. The main split contains 448 samples, designed to challenge LLMs with expert-level knowledge and complex reasoning.

## Motivation

GPQA is one of the most challenging QA datasets for LLMs, requiring expert knowledge and reasoning abilities. As of 2024.10.21, most models (except for the OpenAI O1 model, which is not publicly available) have not saturated performance on this dataset. It remains uncertain whether prompt optimization can effectively enhance an LLM's reasoning capabilities, making GPQA an essential benchmark to evaluate the potential of prompt optimization in reasoning-heavy tasks.

## Preprocessing

1. Shuffle the `gpqa_main` split dataset using seed `42`.
2. Select the first 100 samples as the training set and the next 100 samples as the validation set.

   - For certain biology tasks involving genetic strings (where LLMs tend to generate the genetic string iteratively), select data from `[100:201]` and remove one such instance related to genetic strings.

3. Use the `question` column as the input and format the `Explanation` column and `Correct Answer` column as outputs in the following structure:

   ```json
   {
       "thought": item["Explanation"],
       "answer": item["Correct Answer"]
   }
   ```

## Evaluation

The LLM evaluation is conducted using the prompt below:

```markdown
YOU ARE one of the GREATEST scientists. You are intelligent and rational. You are prudent and cautious. Your mastery over science is unparalleled. You THINK NATURAL, BROAD AND DEEP. Let's think step by step.
Your job is to judge whether the "final_answer" is correct based on "ground_truth_answer", do not be strict on the format, but check the content. Notice that unsolved half results are not Correct.
Question: {question_content}
Is the final_answer correct, given the ground truth answer? Reply with Correct, Wrong or Unknown.
"final_answer": "{final_answer}", "ground_truth_answer": "{ground_truth_answer}"

You MUST respond in JSON format like below:
{{
    "thought": "...",
    "correctness": "<correctness>", One of "Correct", "Wrong", "Unknown"
}}
```
