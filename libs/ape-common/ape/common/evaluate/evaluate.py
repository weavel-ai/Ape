import sys
import tqdm
import asyncio
import pandas as pd
from typing import Any, Dict, List, Optional, Tuple, Union

from ape.common.metric import BaseMetric
from ape.common.global_metric import BaseGlobalMetric, AverageGlobalMetric
from ape.common.types import EvaluationResult, GlobalMetricResult
from ape.common.types.dataset_item import DatasetItem
from ape.common.utils import logger
from ape.common.prompt import Prompt
from .eval_config import EvaluationConfig

try:
    from IPython.display import HTML
    from IPython.display import display as ipython_display
except ImportError:
    ipython_display = print

    def HTML(x) -> str:  # noqa: N802
        return x


from concurrent.futures import ThreadPoolExecutor


# TODO: Counting failures and having a max_failure count. When that is exceeded (also just at the end),
# we print the number of failures, the first N examples that failed, and the first N exceptions raised.
class AsyncExecutor:
    def __init__(self, max_workers):
        self.executor = ThreadPoolExecutor(max_workers=max_workers)
        self.loop = asyncio.get_event_loop()

    async def run_in_executor(self, func, *args):
        return await self.loop.run_in_executor(self.executor, func, *args)

    def shutdown(self):
        self.executor.shutdown()


class Evaluate:
    def __init__(
        self,
        testset: List[DatasetItem],
        metric: BaseMetric,
        global_metric: Optional[BaseGlobalMetric] = None,
        display_progress: bool = False,
        display_table: Union[bool, int] = False,
        max_errors: int = 15,
        return_outputs: bool = False,
        return_global_metric_metadata: Optional[bool] = False,
        batch_size: int = 50,
        **kwargs,
    ):
        self.config = EvaluationConfig(
            testset=testset,
            metric=metric,
            global_metric=global_metric or AverageGlobalMetric(),
            display_progress=display_progress,
            display_table=display_table,
            max_errors=max_errors,
            return_outputs=return_outputs,
            return_global_metric_metadata=return_global_metric_metadata,
            batch_size=batch_size,
            **kwargs,
        )
        self.error_count = 0
        self.total_score = 0

    async def __call__(
        self,
        prompt: Prompt,
        metric: Optional[BaseMetric] = None,
        global_metric: Optional[BaseGlobalMetric] = None,
        testset: Optional[List[DatasetItem]] = None,
        **kwargs,
    ) -> Union[float, Tuple[float, List[EvaluationResult]], Tuple[float, List[float]]]:
        config = self._update_config(metric, global_metric, testset, **kwargs)
        self.total_score = 0
        results: List[EvaluationResult] = await self._process_testset(prompt, config)
        global_result = await self._compute_global_metric(results, config)
        return self._prepare_output(results, global_result, config)

    def _update_config(
        self,
        metric: Optional[BaseMetric],
        global_metric: Optional[BaseGlobalMetric] = None,
        testset: Optional[List[DatasetItem]] = None,
        **kwargs,
    ) -> EvaluationConfig:
        return self.config.model_copy(
            update={
                "metric": metric or self.config.metric,
                "global_metric": global_metric or self.config.global_metric,
                "testset": testset or self.config.testset,
                **kwargs,
            }
        )

    async def _process_testset(
        self, prompt: Prompt, config: EvaluationConfig
    ) -> List[EvaluationResult]:
        async def process_item(example: DatasetItem) -> EvaluationResult:
            try:
                inputs = example.get("inputs", {})
                outputs = example.get("outputs", {})

                prediction = await prompt(**inputs)
                if not prediction:
                    raise ValueError("Prediction is None")
                result = await config.metric(
                    inputs=inputs, gold=outputs, pred=prediction, trace=None
                )
                result = EvaluationResult(
                    example=example,
                    prediction=prediction,
                    score=result.score,
                    intermediate_values=result.intermediate_values,
                )
                return result
            except Exception as e:
                self._handle_error(e, config)
                return EvaluationResult(example=example, prediction={}, score=0.0)

        with tqdm.tqdm(
            total=len(config.testset),
            disable=not config.display_progress,
            file=sys.stdout,
            position=0,
            leave=True,
        ) as pbar:
            tasks = [
                self._bounded_process_item(process_item, item, pbar, config)
                for item in config.testset
            ]
            results: List[EvaluationResult] = await asyncio.gather(*tasks)
            return results

    async def _bounded_process_item(self, process_func, item, pbar, config):
        async with asyncio.Semaphore(config.batch_size):
            result = await process_func(item)
            self._update_progress(pbar, result.score)
            return result

    def _update_progress(self, pbar, score: float):
        self.total_score += score
        pbar.n += 1
        average_score = self.total_score / pbar.n
        pbar.set_description(f"Average Metric: {average_score:.2f} ({100 * average_score:.1f}%)")
        pbar.refresh()

    def _handle_error(self, error: Exception, config: EvaluationConfig):
        self.error_count += 1
        if self.error_count >= config.max_errors:
            raise error
        logger.error(f"Error processing example: {error}")

    async def _compute_global_metric(
        self, results: List[EvaluationResult], config: EvaluationConfig
    ) -> GlobalMetricResult:
        return await config.global_metric(results)

    def _prepare_output(
        self,
        results: List[EvaluationResult],
        global_result: GlobalMetricResult,
        config: EvaluationConfig,
    ) -> Union[
        Tuple[float, List[EvaluationResult], Optional[Dict[str, Any]]],
        Tuple[float, Optional[Dict[str, Any]]],
        Tuple[float, List[EvaluationResult]],
        Tuple[float, List[float]],
        float,
    ]:
        if config.display_table:
            self._display_results_table(results, config)

        if config.return_outputs and config.return_global_metric_metadata:
            return global_result.score, results, global_result.metadata
        elif config.return_global_metric_metadata:
            return global_result.score, global_result.metadata
        elif config.return_outputs:
            return global_result.score, results
        elif config.return_all_scores:
            return global_result.score, [r.score for r in results]
        # elif config.custom:
        #     return global_result, config.calculate(results)
        else:
            return global_result.score

    def _display_results_table(self, results: List[EvaluationResult], config: EvaluationConfig):
        df = pd.DataFrame(
            [
                merge_dicts(
                    r.example if isinstance(r.example, dict) else {"example": r.example},
                    {"prediction": r.prediction, "correct": r.score}
                    | (r.prediction if isinstance(r.prediction, dict) else {}),
                )
                for r in results
            ]
        )
        df = df.map(truncate_cell) if hasattr(df, "map") else df.applymap(truncate_cell)
        df = df.rename(columns={"correct": config.metric.__class__.__name__})

        if isinstance(config.display_table, bool):
            styled_df = configure_dataframe_display(df, config.metric.__class__.__name__)
            truncated_rows = 0
        else:
            styled_df = configure_dataframe_display(
                df.head(config.display_table), config.metric.__class__.__name__
            )
            truncated_rows = len(df) - config.display_table

        ipython_display(styled_df)

        if truncated_rows > 0:
            message = f"""
            <div style='text-align: center; font-size: 16px; font-weight: bold; color: #555; margin: 10px 0;'>
                ... {truncated_rows} more rows not displayed ...
            </div>
            """
            ipython_display(HTML(message))


def merge_dicts(d1: Dict, d2: Dict) -> dict:
    merged = {}
    for k, v in d1.items():
        if k in d2:
            merged[f"example_{k}"] = v
        else:
            merged[k] = v

    for k, v in d2.items():
        if k in d1:
            merged[f"pred_{k}"] = v
        else:
            merged[k] = v

    return merged


def truncate_cell(content) -> str:
    """Truncate content of a cell to 25 words."""
    words = str(content).split()
    if len(words) > 25:
        return " ".join(words[:25]) + "..."
    return content


def configure_dataframe_display(df, metric_name) -> pd.DataFrame:
    """Set various pandas display options for DataFrame."""
    pd.options.display.max_colwidth = None
    pd.set_option("display.max_colwidth", 20)  # Adjust the number as needed
    pd.set_option("display.width", 400)  # Adjust

    df[metric_name] = df[metric_name].apply(lambda x: f"✔️ [{x}]" if x else str(x))

    # Return styled DataFrame
    return df.style.set_table_styles(
        [
            {"selector": "th", "props": [("text-align", "left")]},
            {"selector": "td", "props": [("text-align", "left")]},
        ],
    ).set_properties(
        **{
            "text-align": "left",
            "white-space": "pre-wrap",
            "word-wrap": "break-word",
            "max-width": "400px",
        },
    )
