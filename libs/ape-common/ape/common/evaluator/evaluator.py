import sys
import tqdm
import asyncio
import pandas as pd
from typing import Any, Dict, List, Optional, Tuple, Union

from ape.common.generator import BaseGenerator, Generator
from ape.common.metric import BaseMetric
from ape.common.global_metric import BaseGlobalMetric, AverageGlobalMetric
from ape.common.types import MetricResult, GlobalMetricResult
from ape.common.types.dataset_item import DatasetItem
from ape.common.utils import logger
from ape.common.prompt import Prompt

try:
    from IPython.display import HTML
    from IPython.display import display as ipython_display
except ImportError:
    ipython_display = print

    def HTML(x) -> str:  # noqa: N802
        return x


class Evaluator:
    def __init__(
        self,
        testset: List[DatasetItem],
        metric: BaseMetric,
        generator: Optional[BaseGenerator] = None,
        global_metric: Optional[BaseGlobalMetric] = None,
        display_progress: Optional[bool] = False,
        display_table: Optional[Union[bool, int]] = False,
        max_errors: Optional[int] = 15,
        batch_size: Optional[int] = 100,
        return_only_score: Optional[bool] = True,
        **kwargs,
    ):
        self.testset = testset
        self.generate = generator or Generator()
        self.metric = metric
        self.global_metric = global_metric or AverageGlobalMetric()
        self.display_progress = display_progress
        self.display_table = display_table
        self.max_errors = max_errors
        self.batch_size = batch_size
        self.return_only_score = return_only_score

        self.error_count = 0
        self.total_score = 0

    async def __call__(
        self,
        prompt: Prompt,
        testset: Optional[List[DatasetItem]] = None,
        disply_progress: Optional[bool] = None,
        display_table: Optional[Union[bool, int]] = None,
        max_errors: Optional[int] = None,
        batch_size: Optional[int] = None,
        return_only_score: Optional[bool] = None,
        **kwargs,
    ) -> Union[float, Tuple[List[Union[Dict, str]], List[MetricResult], GlobalMetricResult]]:
        if disply_progress is None:
            disply_progress = self.display_progress
        if display_table is None:
            display_table = self.display_table
        if max_errors is None:
            max_errors = self.max_errors
        if batch_size is None:
            batch_size = self.batch_size
        if return_only_score is None:
            return_only_score = self.return_only_score

        if testset is None:
            testset = self.testset

        results: List[Tuple[Union[str, Dict[str, Any]], MetricResult]] = (
            await self._process_testset(prompt, testset, disply_progress, batch_size, max_errors)
        )
        predictions = [result[0] for result in results]
        eval_results = [result[1] for result in results]
        global_result: GlobalMetricResult = await self.global_metric(eval_results)

        if display_table not in [False, None]:
            self._display_results_table(testset, predictions, eval_results)

        if return_only_score:
            return global_result.score
        else:
            return predictions, eval_results, global_result

    async def _process_testset(
        self,
        prompt: Prompt,
        testset: List[DatasetItem],
        disply_progress: bool,
        batch_size: int,
        max_errors: int,
    ) -> List[Tuple[Union[str, Dict[str, Any]], MetricResult]]:
        async def process_item(
            example: DatasetItem,
        ) -> Tuple[Union[str, Dict[str, Any]], MetricResult]:
            try:
                inputs = example["inputs"]

                prediction = await self.generate(prompt=prompt, inputs=inputs)
                if not prediction:
                    raise ValueError("Prediction is None")
                result = await self.metric(dataset_item=example, pred=prediction)
                return prediction, result
            except Exception as e:
                logger.error(f"Error processing example: {e}")
                self.error_count += 1
                if self.error_count >= max_errors:
                    raise e
                return "", MetricResult(score=0.0)

        with tqdm.tqdm(
            total=len(testset),
            disable=not disply_progress,
            file=sys.stdout,
            position=0,
            leave=True,
        ) as pbar:
            tasks = [
                self._bounded_process_item(process_item, item, pbar, batch_size) for item in testset
            ]
            results: List[Tuple[Union[str, Dict[str, Any]], MetricResult]] = await asyncio.gather(
                *tasks
            )
            return results

    async def _bounded_process_item(self, process_func, item, pbar, batch_size):
        async with asyncio.Semaphore(batch_size):
            result = await process_func(item)
            self._update_progress(pbar, result[1].score)
            return result

    def _update_progress(self, pbar, score: float):
        self.total_score += score
        pbar.n += 1
        average_score = self.total_score / pbar.n
        pbar.set_description(f"Average Metric: {average_score:.2f} ({100 * average_score:.1f}%)")
        pbar.refresh()

    def _display_results_table(
        self,
        testset: List[DatasetItem],
        predictions: List[Union[str, Dict[str, Any]]],
        results: List[MetricResult],
    ):
        df = pd.DataFrame(
            [
                merge_dicts(
                    {"example": dataset_item},
                    {"prediction": pred, "correct": result.score}
                    | (pred if isinstance(result.prediction, dict) else {}),
                )
                for dataset_item, pred, result in zip(testset, predictions, results)
            ]
        )
        df = df.map(truncate_cell) if hasattr(df, "map") else df.applymap(truncate_cell)
        df = df.rename(columns={"correct": self.metric.__class__.__name__})

        if isinstance(self.display_table, bool):
            styled_df = configure_dataframe_display(df, self.metric.__class__.__name__)
            truncated_rows = 0
        else:
            styled_df = configure_dataframe_display(
                df.head(self.display_table), self.metric.__class__.__name__
            )
            truncated_rows = len(df) - self.display_table

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
