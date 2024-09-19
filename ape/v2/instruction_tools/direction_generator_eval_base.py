import asyncio
import json
from typing import Dict
from ape.prompt.prompt_base import Prompt
from ape.types import GlobalMetricResult

async def direction_generator_eval_base(
    task_description: str,
    base_prompt: Prompt,
    trainset_result: GlobalMetricResult,
    metric_description: str,
    tip: str,
) -> str:
    evaluation_result = "final score : " + str(trainset_result.score) + "\nmetadata : " + str(trainset_result.metadata)
    base_prompt_messages = [
        json.dumps(message) for message in base_prompt.messages
    ]
    base_prompt_messages_str = "\n".join(base_prompt_messages)
    
    direction_generator = Prompt.from_filename("direction-generator-eval-base")
    
    for attempt in range(3):
        try:
            print(tip)
            suggestion_str = await direction_generator(
                task_description=task_description,
                base_prompt=base_prompt_messages_str,
                response_format=base_prompt.response_format.model_dump(exclude_none=True) if base_prompt.response_format else None, 
                evaluation_result=evaluation_result,
                metric_description=metric_description,
                tip=tip,
            )
            
            if not suggestion_str.startswith("{"):
                suggestion_str = "{" + suggestion_str
            
            suggestion = json.loads(suggestion_str)
            
            return suggestion["direction_description"]
        except Exception as e:
            if attempt == 2:  # Last attempt
                raise e
            print(f"Error occurred: {e}. Retrying... (Attempt {attempt + 1}/3)")
            await asyncio.sleep(1)  # Wait for 1 second before retrying
    
                