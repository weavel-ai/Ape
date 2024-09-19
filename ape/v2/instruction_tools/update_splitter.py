import asyncio
import json
from typing import Dict, List
from ape.prompt.prompt_base import Prompt
from ape.proposer.utils import extract_prompt

async def update_splitter(
    base_prompt: Prompt,
    new_prompt: Prompt,
    direction_description: str,
) -> List[str]:

    base_prompt_messages = [
        json.dumps(message) for message in base_prompt.messages
    ]
    base_prompt_messages_str = "\n".join(base_prompt_messages)

    new_prompt_messages = [
        json.dumps(message) for message in new_prompt.messages
    ]
    new_prompt_messages_str = "\n".join(new_prompt_messages)

    splitter = Prompt.from_filename("update-splitter")

    for attempt in range(3):
        try:
            splitter_result: str = await splitter(
                base_prompt=base_prompt_messages_str,
                new_prompt=new_prompt_messages_str,
                direction_description=direction_description,
            )
            if splitter_result[0] != "{":
                splitter_result = "{" + splitter_result
            
            splitter_result = json.loads(splitter_result)
            
            splitted_updates = splitter_result.get("updates", [])
            
            return splitted_updates
        except Exception as e:
            if attempt == 2:  # Last attempt
                raise e
            print(f"Error occurred: {e}. Retrying... (Attempt {attempt + 1}/3)")
            await asyncio.sleep(1)  # Wait for 1 second before retrying
    