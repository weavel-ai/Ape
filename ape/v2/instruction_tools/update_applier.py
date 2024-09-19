import asyncio
import json
from typing import Dict
from ape.prompt.prompt_base import Prompt
from ape.proposer.utils import extract_prompt

async def update_applier(
    task_description: str,
    base_prompt: Prompt,
    direction_description: str,
    splitted_update: str,
) -> Prompt:
    base_prompt_messages = [
        json.dumps(message) for message in base_prompt.messages
    ]
    base_prompt_messages_str = "\n".join(base_prompt_messages)
    
    instruction_generator = Prompt.from_filename("gen-instruction-by-splitted-update")
    
    for attempt in range(3):
        try:
            suggested_prompt_str: Dict[str, str] = await instruction_generator(
                task_description=task_description,
                base_prompt=base_prompt_messages_str,
                response_format=base_prompt.response_format.model_dump(exclude_none=True) if base_prompt.response_format else None, 
                direction_description=direction_description,
                update=splitted_update,
            )
            
            if "```prompt" not in suggested_prompt_str:
                suggested_prompt_str = "```prompt\n" + suggested_prompt_str
            extracted_prompt = extract_prompt(suggested_prompt_str)
            
            new_prompt = Prompt.load(extracted_prompt)
            if not new_prompt.messages:
                raise ValueError("Generated prompt has no messages")
            new_prompt.model = base_prompt.model
            new_prompt.response_format = base_prompt.response_format
            new_prompt.name = "InstructNew"

            return new_prompt
        except Exception as e:
            if attempt == 2:  # Last attempt
                raise e
            print(f"Error occurred: {e}. Retrying... (Attempt {attempt + 1}/3)")
            await asyncio.sleep(1)  # Wait for 1 second before retrying
    
                