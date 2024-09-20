import asyncio
import json
from ape.prompt.prompt_base import Prompt
from ape.types import GlobalMetricResult
from ape.types.v2 import ValueAnalysisResult

async def value_analyzer(
    metric_description: str,
    experiment_description: str,
    new_prompt: Prompt,
    base_result: GlobalMetricResult,   
    new_result: GlobalMetricResult,
) -> ValueAnalysisResult:
    new_prompt_messages = [
        json.dumps(message) for message in new_prompt.messages
    ]
    new_prompt_messages_str = "\n".join(new_prompt_messages)
    
    value_analyzer = Prompt.from_filename("value-analyzer")
    
    for attempt in range(3):
        try:
            value_analysis_result_str: str = await value_analyzer(
                metric_description=metric_description,
                experiment_description=experiment_description,
                base_prompt_result=str(base_result.model_dump()),
                new_prompt=new_prompt_messages_str,
                new_prompt_result=str(new_result.model_dump()),
            )
            
            if not value_analysis_result_str.startswith("{"):
                value_analysis_result_str = "{" + value_analysis_result_str
            value_analysis_result = json.loads(value_analysis_result_str)
            
            return ValueAnalysisResult(
                think=value_analysis_result["think"],
                is_success=value_analysis_result["is_success"],
                score=value_analysis_result["score"],
            )
        except Exception as e:
            if attempt == 2:  # Last attempt
                raise e
            print(f"Error occurred: {e}. Retrying... (Attempt {attempt + 1}/3)")
            await asyncio.sleep(1)  # Wait for 1 second before retrying
    
                