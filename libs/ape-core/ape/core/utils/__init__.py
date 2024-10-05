import asyncio
import re
import random

from ape.common.prompt import Prompt
from ape.common.types import ResponseFormat
from ape.common.prompt import Prompt
from ape.common.utils import logger
from ape.core.core_prompts import ApeCorePrompts

_nest_asyncio_applied = False


def extract_prompt(text: str) -> str:
    match = re.search(r"```prompt(.*?)```", text, re.DOTALL)
    if match:
        return match.group(1).strip()
    else:
        raise ValueError("No prompt found")


def get_response_format_instructions(response_format: ResponseFormat):
    if response_format["type"] == "json_object":
        return (
            "The prompt should enforce a JSON output and must include the word JSON in the prompt."
        )
    else:
        return ""


async def reformat_prompt(prompt: Prompt, response_format: ResponseFormat) -> Prompt:
    """Reformat the prompt to be in XML style."""
    formatter_filename: str
    match response_format["type"]:
        case "json_object":
            formatter_filename = "reformat-prompt-json-object"
        case "json_schema":
            formatter_filename = "reformat-prompt-json-schema"
    formatter = ApeCorePrompts.get(formatter_filename)
    new_prompt: Prompt
    retry_count = 0
    logger.info(f"Reformatting prompt: {prompt.dump()}")
    while True:
        try:
            res = await formatter(prompt=prompt.dump())
            extracted = extract_prompt(res)
            if response_format["type"] == "json_object":
                if "json" not in extracted.lower():
                    raise ValueError("Reformatted prompt does not include the word 'JSON'")
            logger.info(f"Reformatted prompt: {extracted}")
            new_prompt_message = Prompt.load(extracted)
            new_prompt = prompt.deepcopy()
            new_prompt.messages = new_prompt_message.messages

            break
        except Exception as e:
            logger.error(f"Error reformatting prompt: {e}. Retrying...")
            retry_count += 1
            if retry_count > 10:
                logger.error("Failed to reformat prompt after 3 retries")
                logger.error("Generated prompt:" + res)
                raise e

    # new_prompt.fewshot_config = prompt.fewshot_config # TODO: fix this more pretty way
    return new_prompt


def run_async(coroutine):
    global _nest_asyncio_applied

    if not _nest_asyncio_applied:
        try:
            import nest_asyncio

            nest_asyncio.apply()
            _nest_asyncio_applied = True
        except ImportError:
            logger.error("Please install nest_asyncio: !pip install nest_asyncio")
            raise

    try:
        loop = asyncio.get_event_loop()
        if loop.is_running():
            # If the loop is already running, create a new task and wait for it to complete
            task = asyncio.ensure_future(coroutine)
            return loop.run_until_complete(task)
        else:
            # If the loop is not running, use run_until_complete directly
            return loop.run_until_complete(coroutine)
    except RuntimeError:
        # If no event loop is present, use asyncio.run
        return asyncio.run(coroutine)


def create_minibatch(trainset, batch_size=50):
    """Create a minibatch from the trainset."""

    # Ensure batch_size isn't larger than the size of the dataset
    batch_size = min(batch_size, len(trainset))

    # Randomly sample indices for the mini-batch
    sampled_indices = random.sample(range(len(trainset)), batch_size)

    # Create the mini-batch using the sampled indices
    minibatch = [trainset[i] for i in sampled_indices]

    return minibatch
