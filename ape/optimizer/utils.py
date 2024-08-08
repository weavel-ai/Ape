import asyncio
import logging
import os
import random
from typing import Any, Awaitable, Callable, Dict, List, Optional

import numpy as np
from ape.optimizer.bootstrap_fewshot import BootstrapFewShot
from ape.optimizer.fewshot_optimizer import FewShotOptimizer
from ape.prompt.prompt_base import Prompt
from ape.proposer.utils import extract_prompt
from ape.types import Dataset


async def reformat_prompt_xml_style(prompt: Prompt) -> Prompt:
    """Reformat the prompt to be in XML style."""
    formatter = Prompt.from_filename("reformat-prompt-xml-style")
    new_prompt: Prompt
    retry_count = 0
    while True:
        try:
            res = await formatter(prompt=prompt.dump())
            extracted = extract_prompt(res)
            logging.info(f"Reformatted prompt: {extracted}")
            new_prompt = Prompt.load(extracted)
            break
        except Exception as e:
            logging.error(f"Error reformatting prompt: {e}. Retrying...")
            retry_count += 1
            if retry_count > 3:
                logging.error("Failed to reformat prompt after 3 retries")
                raise e
    return new_prompt


async def create_single_fewshot_demo_set(
    student: Prompt,
    trainset: Dataset,
    seed: int,
    max_labeled_demos: int,
    max_bootstrapped_demos: int,
    metric: Callable[..., Awaitable[Any]],
    teacher_settings: dict,
    max_rounds: int,
    labeled_sample: bool,
    min_num_samples: int,
    metric_threshold: Any,
    teacher: Any,
    include_non_bootstrapped: bool,
) -> Prompt:
    trainset2 = list(trainset)

    if seed == -3 and include_non_bootstrapped:
        # zero-shot
        prompt2 = student.reset_copy()
    elif seed == -2 and max_labeled_demos > 0 and include_non_bootstrapped:
        # labels only
        optimizer = FewShotOptimizer(k=max_labeled_demos)
        prompt2 = await optimizer.optimize(
            student, trainset=trainset2, sample=labeled_sample
        )
    elif seed == -1:
        # unshuffled few-shot
        prompt = BootstrapFewShot(
            metric=metric,
            max_bootstrapped_demos=max_bootstrapped_demos,
            max_labeled_demos=max_labeled_demos,
            teacher_settings=teacher_settings,
            max_rounds=max_rounds,
        )
        prompt2 = await prompt.optimize(student, teacher=teacher, trainset=trainset2)
    else:
        # shuffled few-shot
        random.Random(seed).shuffle(trainset2)
        size = random.Random(seed).randint(min_num_samples, max_bootstrapped_demos)
        optimizer = BootstrapFewShot(
            metric=metric,
            metric_threshold=metric_threshold,
            max_bootstrapped_demos=size,
            max_labeled_demos=max_labeled_demos,
            teacher_settings=teacher_settings,
            max_rounds=max_rounds,
        )
        prompt2 = await optimizer.optimize(student, teacher=teacher, trainset=trainset2)

    return prompt2


async def create_n_fewshot_demo_sets(
    student: Prompt,
    num_candidate_sets: int,
    trainset: Dataset,
    max_labeled_demos: int,
    max_bootstrapped_demos: int,
    metric: Callable[..., Awaitable[Any]],
    teacher_settings: dict,
    max_rounds=1,
    labeled_sample=True,
    min_num_samples=1,
    metric_threshold=None,
    teacher=None,
    include_non_bootstrapped=True,
    seed=0,
) -> List[Dataset]:
    num_candidate_sets -= 3
    random.Random(seed).shuffle(trainset)

    tasks = []
    for seed in range(-3, num_candidate_sets):
        task = create_single_fewshot_demo_set(
            student=student,
            trainset=trainset,
            seed=seed,
            max_labeled_demos=max_labeled_demos,
            max_bootstrapped_demos=max_bootstrapped_demos,
            metric=metric,
            teacher_settings=teacher_settings,
            max_rounds=max_rounds,
            labeled_sample=labeled_sample,
            min_num_samples=min_num_samples,
            metric_threshold=metric_threshold,
            teacher=teacher,
            include_non_bootstrapped=include_non_bootstrapped,
        )
        tasks.append(task)

    fewshot_candidates = await asyncio.gather(*tasks)
    return [prompt.fewshot for prompt in fewshot_candidates]


def create_minibatch(trainset, batch_size=50):
    """Create a minibatch from the trainset."""

    # Ensure batch_size isn't larger than the size of the dataset
    batch_size = min(batch_size, len(trainset))

    # Randomly sample indices for the mini-batch
    sampled_indices = random.sample(range(len(trainset)), batch_size)

    # Create the mini-batch using the sampled indices
    minibatch = [trainset[i] for i in sampled_indices]

    return minibatch


async def eval_candidate_prompt(
    batch_size: int,
    trainset: Dataset,
    candidate_prompt: Prompt,
    evaluate: Callable[..., Awaitable[Any]],
):
    """Evaluate a candidate program on the trainset, using the specified batch size."""
    # Evaluate on the full trainset
    if batch_size >= len(trainset):
        score = await evaluate(candidate_prompt, devset=trainset, display_table=0)
    # Or evaluate on a minibatch
    else:
        score = await evaluate(
            candidate_prompt,
            devset=create_minibatch(trainset, batch_size),
            display_table=0,
        )

    return score


def save_candidate_prompt(
    prompt: Prompt, log_dir: Optional[str], trial_num: int, note=None
):
    """Save the candidate prompt to the log directory."""

    if log_dir is None:
        return None

    # Ensure the directory exists
    eval_programs_dir = os.path.join(log_dir, "evaluated_prompts")
    os.makedirs(eval_programs_dir, exist_ok=True)

    # Define the save path for the program
    if note:
        save_path = os.path.join(eval_programs_dir, f"prompt_{trial_num}_{note}.prompt")
    else:
        save_path = os.path.join(eval_programs_dir, f"prompt_{trial_num}.prompt")

    # Save the prompt
    with open(save_path, "w") as f:
        f.write(prompt.dump())
    return save_path


def get_prompt_with_highest_avg_score(
    param_score_dict: Dict, fully_evaled_param_combos: Dict
):
    """Used as a helper function for bayesian + minibatching optimizers. Returns the program with the highest average score from the batches evaluated so far."""

    # Calculate the mean for each combination of categorical parameters, based on past trials
    results = []
    for key, values in param_score_dict.items():
        scores = np.array([v[0] for v in values])
        mean = np.average(scores)
        program = values[0][1]
        results.append((key, mean, program))

    # Sort results by the mean
    sorted_results = sorted(results, key=lambda x: x[1], reverse=True)

    # Find the combination with the highest mean, skip fully evaluated ones
    for combination in sorted_results:
        key, mean, program = combination

        if key in fully_evaled_param_combos:
            continue

        print(f"Best Combination: {key} with Mean = {mean}")

        return program, key

    # If no valid program is found, we return the last valid one that we found
    return program, key
