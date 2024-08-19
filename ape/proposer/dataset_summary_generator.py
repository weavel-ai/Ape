from ape.prompt.prompt_base import Prompt
from ape.types import Dataset
from ape.types.dataset_item import DatasetItem
from ape.types.response_format import ResponseFormat


async def process_batch(
    trainset: Dataset,
    batch_start: int,
    batch_end: int,
    prompt_model: str,
) -> str:
    """
    Process a batch of the dataset to generate observations.

    Args:
        trainset (Dataset): The full training dataset.
        batch_start (int): The starting index of the batch.
        batch_end (int): The ending index of the batch.
        prompt_model (str): The name of the prompt model to use.

    Returns:
        str: The generated observations for the batch.
    """
    descriptor = Prompt.from_filename("dataset-descriptor")
    descriptor.response_format = ResponseFormat(type="xml")
    descriptor.model = prompt_model
    trainset = [
        item.model_dump() if isinstance(item, DatasetItem) else item
        for item in trainset[batch_start:batch_end]
    ]
    output = await descriptor(
        lm_config=dict(temperature=1.0),
        examples=", ".join([sorted(item.keys()).__repr__() for item in trainset]),
    )
    return output["observations"]


async def create_dataset_summary(
    trainset: Dataset,
    view_data_batch_size: int,
    prompt_model: str,
) -> str:
    """
    Create a summary of the dataset by processing a batch of it.

    Args:
        trainset (Dataset): The full training dataset.
        view_data_batch_size (int): The maximum number of items to process.
        prompt_model (str): The name of the prompt model to use.

    Returns:
        str: The generated observations summarizing the dataset.
    """
    upper_lim = min(len(trainset), view_data_batch_size)

    # Initial batch processing
    observations = await process_batch(trainset, 0, upper_lim, prompt_model)

    return observations
