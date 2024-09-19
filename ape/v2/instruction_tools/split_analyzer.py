import asyncio
import json
from typing import List,Tuple
from ape.prompt.prompt_base import Prompt
from ape.types import GlobalMetricResult
from ape.types.v2 import SplitAnalysisResult

async def split_analyzer(
    experiment_description: str,
    splitted_results: List[Tuple[str, GlobalMetricResult]],
) -> SplitAnalysisResult:
    
    split_analyzer = Prompt.from_filename("split-analyzer")
    
    experiment_results = ""
    for i, (update, result) in enumerate(splitted_results):
        experiment_results += f"--- {i+1}th Experiment Result ---\n"
        experiment_results += f"Prompt Update: {update}\n"
        experiment_results += "Result: "
        experiment_results += f"Score: {result.score}\n"
        experiment_results += f"Metadata: {json.dumps(result.metadata)}\n"
    
    for attempt in range(3):
        try:
            split_analysis_result_str: str = await split_analyzer(
                experiment_description=experiment_description,
                experiment_results=experiment_results,
            )
            
            if not split_analysis_result_str.startswith("{"):
                split_analysis_result_str = "{" + split_analysis_result_str
            split_analysis_result = json.loads(split_analysis_result_str)
            
            return SplitAnalysisResult(
                **split_analysis_result
            )
            
        except Exception as e:
            if attempt == 2:  # Last attempt
                raise e
            print(f"Error occurred: {e}. Retrying... (Attempt {attempt + 1}/3)")
            await asyncio.sleep(1)  # Wait for 1 second before retrying
    
                