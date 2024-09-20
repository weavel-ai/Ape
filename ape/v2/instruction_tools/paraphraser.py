import asyncio
import json
from typing import List, Tuple
from ape.prompt.prompt_base import Prompt
from ape.proposer.utils import extract_prompt

async def paraphraser(
    experiment_description: str,
    base_prompt: Prompt,
    new_prompt: Prompt,
    previous_paraphrase_results: str,
) -> Tuple[Prompt, str]:
    base_prompt_messages = [
        json.dumps(message) for message in base_prompt.messages
    ]
    base_prompt_messages_str = "\n".join(base_prompt_messages)
    
    new_prompt_messages = [
        json.dumps(message) for message in new_prompt.messages
    ]
    new_prompt_messages_str = "\n".join(new_prompt_messages)
    
    paraphraser = Prompt.from_filename("update-paraphraser")
    extractor = Prompt.from_filename("update-extractor")
    
    for attempt in range(3):
        try:
            paraphrased_prompt_str: str = await paraphraser(
                experiment_description=experiment_description,
                base_prompt=base_prompt_messages_str,
                new_prompt=new_prompt_messages_str,
                previous_paraphrase_results=previous_paraphrase_results,
            )
            
            if not paraphrased_prompt_str.startswith('```prompt'):
                paraphrased_prompt_str = '```prompt\n' + paraphrased_prompt_str
            
            extracted_prompt = extract_prompt(paraphrased_prompt_str)
            paraphrased_prompt = Prompt.load(extracted_prompt)
            if not paraphrased_prompt.messages:
                raise ValueError("Generated prompt has no messages")
            paraphrased_prompt.model = base_prompt.model
            paraphrased_prompt.response_format = base_prompt.response_format
            paraphrased_prompt.name = "InstructNew"
            
            paraphrased_prompt_messages = [
                json.dumps(message) for message in paraphrased_prompt.messages
            ]
            paraphrased_prompt_messages_str = "\n".join(paraphrased_prompt_messages)
            
            paraphrased_part = await extractor(
                base_prompt=base_prompt_messages_str,
                new_prompt=paraphrased_prompt_messages_str,
                direction_description=experiment_description,
            )
            
            if not paraphrased_part.startswith('{'):
                paraphrased_part = '{' + paraphrased_part
            
            paraphrased_parts = json.loads(paraphrased_part)["updates"]
            paraphrased_part = "\n".join(paraphrased_parts)
            return paraphrased_prompt, paraphrased_part
        
        except Exception as e:
            if attempt == 2:  # Last attempt
                raise e
            print(f"Error occurred: {e}. Retrying... (Attempt {attempt + 1}/3)")
            await asyncio.sleep(1)  # Wait for 1 second before retrying
    
                