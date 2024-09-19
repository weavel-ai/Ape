import asyncio
import json
from typing import Optional
from ape.prompt.prompt_base import Prompt
from ape.types import Dataset, DatasetItem
from ape.types import ResponseFormat

async def dataset_summarizer(
    trainset: Dataset,
    view_data_batch_size: Optional[int] = 10,
) -> str:
    upper_lim = min(len(trainset), view_data_batch_size)
    
    descriptor = Prompt.from_filename("dataset-descriptor")
    
    for attempt in range(3):
        try:
            formatted_examples = format_examples(trainset, 0, upper_lim)
            
            output = await descriptor(
                lm_config=dict(temperature=1.0),
                examples=formatted_examples,
            )
            
            res = ""
            for line in output["observations"]:
                res += line + "\n"
            return res
        except Exception as e:
            if attempt == 2:  # Last attempt
                raise e
            print(f"Error occurred: {e}. Retrying... (Attempt {attempt + 1}/3)")
            await asyncio.sleep(1)  # Wait for 1 second before retrying

def format_examples(trainset: Dataset, batch_start: int, batch_end: int) -> str:
    formatted_examples = ""
    for idx, item in enumerate(trainset[batch_start:batch_end]):
        formatted_examples += f"### Demo {idx+1} ###\n"
        
        if isinstance(item, DatasetItem):
            inputs, outputs = item.inputs, item.outputs
        else:
            inputs, outputs = item.get("inputs", {}), item.get("outputs", {})
        
        formatted_examples += "**Inputs**\n"
        for key, value in inputs.items():
            formatted_examples += f"{key.capitalize()}:\n{value}\n"
        
        formatted_examples += f"**Outputs**\n{json.dumps(outputs, indent=2)}\n\n"
    
    return formatted_examples
