# New York Times Topics

## Source

[dstefan/New_York_Times_Topics](https://huggingface.co/datasets/dstefa/New_York_Times_Topics)

## Description

The **New York Times Topics** dataset is a topic classification dataset consisting of New York Times articles. It contains 256,000 training samples and is designed for multi-label classification tasks, where each article may belong to multiple topics.

## Motivation

We aim to evaluate prompt optimization performance on straightforward multi-label classification tasks. The New York Times Topics dataset is ideal for this purpose because it does not fully saturate the performance of large language models (LLMs). In our tests with **gpt-4o-mini**, we found that the model's performance reached approximately 75%.

If a more challenging simple classification task is identified, we will discontinue using this dataset as a benchmark for classification tasks in the Ape project.

## Preprocessing

1. Shuffle the dataset using seed `1`.
2. Select the first 100 samples (`[:100]`) as the training set and samples from index 200 to 300 (`[200:300]`) as the validation set.
3. Rename the `article` column to `text` for inputs, and format the `topic_name` column as outputs using the structure: `{"topics": "..."}`.
4. Add the following information about the candidate topics to the prompt:

```markdown
These are the list of candidate topics:
['Politics', 'Crime', 'Lifestyle and Fashion', 'Science and Technology', 'Arts, Culture, and Entertainment', 'Sports', 'Business and Finance', 'Health and Wellness']
```

## Evaluation

The evaluation is based on an exact match between the predicted topic and the ground truth topic.
