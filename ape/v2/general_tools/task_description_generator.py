import asyncio
import json
from typing import Optional
from ape.prompt.prompt_base import Prompt
from ape.types import Dataset
from ape.types.response_format import ResponseFormat
from ape.v2.general_tools.dataset_summarizer import dataset_summarizer

async def task_description_generator(
    trainset: Dataset,
    base_prompt: Prompt,
) -> str:
    """
    Generate a task description based on the dataset description, base prompt, and trainset.

    Args:
        dataset_description (str): A description of the dataset.
        base_prompt (Optional[Prompt]): The base prompt to use, if any.
        trainset (Dataset): The training dataset.

    Returns:
        str: The generated task description.
    """
    describe_prompt = Prompt.from_filename("describe-prompt")
    
    base_prompt_messages = [
        json.dumps(message) for message in base_prompt.messages
    ]
    base_prompt_messages_str = "\n".join(base_prompt_messages)
    
    for attempt in range(3):
        try:
            # Generate dataset summary
            dataset_summary = await dataset_summarizer(trainset)
            
            # Describe the prompt
            prompt_description = await describe_prompt(prompt=base_prompt_messages_str, dataset_description=dataset_summary)
            
            prompt_description = prompt_description["description"]
            
            return prompt_description
        except Exception as e:
            if attempt == 2:  # Last attempt
                print(f"Error generating task description: {e}")
                return ""
            print(f"Error occurred: {e}. Retrying... (Attempt {attempt + 1}/3)")
            await asyncio.sleep(1)  # Wait for 1 second before retrying
