import asyncio
import json
import inspect
from ape.prompt.prompt_base import Prompt
from ape.metric import (
    CosineSimilarityMetric,
    JsonMatchMetric,
    SemanticF1Metric,
    PydanticMatchMetric,
)
from ape.utils import logger

async def metric_description_generator(metric, global_metric=None) -> str:
    """
    Generate a description of the metric.
    """
    for attempt in range(3):
        try:
            if isinstance(metric, CosineSimilarityMetric):
                metric_str = "Measures how similar the predicted text is to the correct answer by comparing their vector representations. A higher score means the prediction is more similar to the gold."
            elif isinstance(metric, JsonMatchMetric):
                metric_str = "Compares JSON format predictions with the correct JSON answer. It checks if each key-value pair in the prediction matches the ground truth exactly. The score reflects how many pairs match correctly."
            elif isinstance(metric, SemanticF1Metric):
                metric_str = """\
        Evaluates how well the prediction captures the meaning of the correct answer:
        1. Extracts key statements from both the prediction and ground truth.
        2. Checks how many statements from the prediction are found in the ground truth (Precision).
        3. Checks how many statements from the ground truth are found in the prediction (Recall).
        4. Calculates the F1 score, which balances Precision and Recall. A higher score indicates better semantic matching."""
            elif isinstance(metric, PydanticMatchMetric):
                metric_str = "Compares structured data (in JSON format) between the prediction and the correct answer. It checks if each field and its value in the prediction matches the ground truth exactly. The score indicates how closely the prediction's structure matches the expected format."
            else:
                compute_function = getattr(metric, "compute", None)
                compute_function_source_code = inspect.getsource(compute_function)
                
                if global_metric:
                    global_metric_compute_function = getattr(global_metric, "compute", None)
                    global_metric_compute_function_source_code = inspect.getsource(global_metric_compute_function)
                    
                    # get Prompt gen-metric-description-with-global-metric.prompt
                    gen_metric_description_with_global_metric = Prompt.from_filename("gen-metric-description-with-global-metric")
                    metric_str = await gen_metric_description_with_global_metric(
                        metric_sourcecode=compute_function_source_code,
                        global_metric_sourcecode=global_metric_compute_function_source_code,
                    )
                else:
                    gen_metric_description = Prompt.from_filename("gen-metric-description")
                    metric_str = await gen_metric_description(
                        metric_sourcecode=compute_function_source_code,
                    )
            return metric_str
        except Exception as e:
            if attempt == 2:  # Last attempt
                logger.error(f"Error generating metric description: {e}")
                return ""
            print(f"Error occurred: {e}. Retrying... (Attempt {attempt + 1}/3)")
            await asyncio.sleep(1)  # Wait for 1 second before retrying