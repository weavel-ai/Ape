import asyncio
import json
from typing import List
from ape.prompt.prompt_base import Prompt
from ape.proposer.utils import extract_prompt

async def paraphraser(
    experiment_description: str,
    base_prompt: Prompt,
    new_prompt: Prompt,
    update: str,
) -> List[str]:
    base_prompt_messages = [
        json.dumps(message) for message in base_prompt.messages
    ]
    base_prompt_messages_str = "\n".join(base_prompt_messages)
    
    new_prompt_messages = [
        json.dumps(message) for message in new_prompt.messages
    ]
    new_prompt_messages_str = "\n".join(new_prompt_messages)
    
    paraphraser = Prompt.from_filename("update-paraphraser")
    
    for attempt in range(3):
        try:
            paraphrased_prompt_str: str = await paraphraser(
                experiment_description=experiment_description,
                base_prompt=base_prompt_messages_str,
                new_prompt=new_prompt_messages_str,
                update=update,
            )
            
            if not paraphrased_prompt_str.startswith("{"):
                paraphrased_prompt_str = "{" + paraphrased_prompt_str
            
            paraphrased_prompt = json.loads(paraphrased_prompt_str)
            
            new_updates = paraphrased_prompt.get("updates", [])
            return new_updates
        except Exception as e:
            if attempt == 2:  # Last attempt
                raise e
            print(f"Error occurred: {e}. Retrying... (Attempt {attempt + 1}/3)")
            await asyncio.sleep(1)  # Wait for 1 second before retrying
    
                