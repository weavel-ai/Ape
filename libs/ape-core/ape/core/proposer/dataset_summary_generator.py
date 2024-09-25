import json
from typing import List
from ape.common.types import DatasetItem, ResponseFormat

from ape.core.core_prompts import ApeCorePrompts


async def process_batch(
    trainset: List[DatasetItem],
    batch_start: int,
    batch_end: int,
) -> str:
    """
    Process a batch of the dataset to generate observations.

    Args:
        trainset (List[DatasetItem]): The full training dataset.
        batch_start (int): The starting index of the batch.
        batch_end (int): The ending index of the batch.

    Returns:
        str: The generated observations for the batch.
    """
    descriptor = ApeCorePrompts.get("dataset-descriptor")

    # Format examples using a similar approach to format_fewshot
    formatted_examples = ""
    for idx, item in enumerate(trainset[batch_start:batch_end]):
        formatted_examples += f"### Demo {idx+1} ###\n"

        inputs, outputs = item.get("inputs", {}), item.get("outputs", {})

        formatted_examples += "**Inputs**\n"
        for key, value in inputs.items():
            formatted_examples += f"{key.capitalize()}:\n{value}\n"

        formatted_examples += f"**Outputs**\n{json.dumps(outputs, indent=2)}\n\n"

    output = await descriptor(
        lm_config=dict(temperature=1.0),
        examples=formatted_examples,
    )
    return output["observations"]


async def create_dataset_summary(
    trainset: List[DatasetItem],
    view_data_batch_size: int,
) -> str:
    """
    Create a summary of the dataset by processing a batch of it.

    Args:
        trainset (List[DatasetItem]): The full training dataset.
        view_data_batch_size (int): The maximum number of items to process.

    Returns:
        str: The generated observations summarizing the dataset.
    """
    upper_lim = min(len(trainset), view_data_batch_size)

    # Initial batch processing
    observations = await process_batch(trainset, 0, upper_lim)

    return observations
