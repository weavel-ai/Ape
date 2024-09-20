import asyncio
import json
from typing import List,Tuple
from ape.prompt.prompt_base import Prompt
from ape.types import GlobalMetricResult
from ape.types.v2 import ParaphraseAnalysisResult, ValueAnalysisResult

async def paraphrase_analyzer(
    experiment_description: str,
    paraphrase_results: List[Tuple[str, GlobalMetricResult, ValueAnalysisResult]],
) -> ParaphraseAnalysisResult:
    
    paraphrase_analyzer = Prompt.from_filename("paraphrase-analyzer")
    
    paraphrase_results_str = ""
    for i, (update, result, value) in enumerate(paraphrase_results):
        paraphrase_results_str += f"--- {i+1}th Experiment Result ---\n"
        paraphrase_results_str += f"Prompt Update: {update}\n"
        paraphrase_results_str += "Result: "
        paraphrase_results_str += f"Score: {result.score}\n"
        paraphrase_results_str += f"Metadata: {json.dumps(result.metadata)}\n"
        paraphrase_results_str += f"Experiment Success: {value.is_success}\n"
    
    for attempt in range(3):
        try:
            paraphrase_analysis_result_str: str = await paraphrase_analyzer(
                experiment_description=experiment_description,
                paraphrase_results=paraphrase_results_str,
            )
            
            if not paraphrase_analysis_result_str.startswith("{"):
                paraphrase_analysis_result_str = "{" + paraphrase_analysis_result_str
            paraphrase_analysis_result = json.loads(paraphrase_analysis_result_str)
            
            return ParaphraseAnalysisResult(
                **paraphrase_analysis_result
            )
            
        except Exception as e:
            if attempt == 2:  # Last attempt
                raise e
            print(f"Error occurred: {e}. Retrying... (Attempt {attempt + 1}/3)")
            await asyncio.sleep(1)  # Wait for 1 second before retrying
    
                