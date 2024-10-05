import asyncio
import json
import random
from typing import Any, Dict, List, Literal, Optional, Tuple
from ape.common.prompt.prompt_base import Prompt
from ape.common.types import GlobalMetricResult, MetricResult, DatasetItem
from ape.common.generate import BaseGenerate
from ape.common.global_metric import BaseGlobalMetric
from ape.common.metric import BaseMetric
from ape.core.core_prompts import ApeCorePrompts
from ape.core.proposer.utils import extract_prompt
from ape.core.v2.trainer.base import BaseTrainer
from ape.core.v2.types.report import ExpelTrainerReport
from ape.core.optimizer.utils import reformat_prompt


class ExpelTrainer(BaseTrainer):
    def __init__(
        self,
        generator: BaseGenerate,
        metric: BaseMetric,
        global_metric: BaseGlobalMetric,
        random_seed: int = 42,
        max_proposals_per_step: int = 5,
        target_subgroup: Literal["success", "failure", "all"] = "all",
        **kwargs,
    ):
        super().__init__(generator, metric, global_metric, **kwargs)
        self.random_seed = random_seed
        self.max_proposals_per_step = max_proposals_per_step
        self.target_subgroup = target_subgroup
        self.success_feedback_generator = ApeCorePrompts.get("expel-success-feedback-generator")
        self.failure_feedback_generator = ApeCorePrompts.get("expel-failure-feedback-generator")
        self.feedback_applier = ApeCorePrompts.get("expel-feedback-applier")

        random.seed(random_seed)

    async def train(
        self,
        prompt: Prompt,
        trainset: List[DatasetItem],
        valset: List[DatasetItem],
    ) -> Tuple[Prompt, ExpelTrainerReport]:
        report = ExpelTrainerReport(scores=[], feedbacks=[], best_score=0.0)
        # embedding_map = await self.create_embedding_map(trainset)

        success_dataset = []
        failure_dataset = []
        success_predictions = []
        failure_predictions = []
        success_eval_results = []
        failure_eval_results = []
        predictions, eval_results, _ = await self._evaluate(dataset=trainset, prompt=prompt)
        for data, pred, eval_result in zip(trainset, predictions, eval_results):
            if eval_result.score == 1.0:
                success_dataset.append(data)
                success_predictions.append(pred)
                success_eval_results.append(eval_result)
            else:
                failure_dataset.append(data)
                failure_predictions.append(pred)
                failure_eval_results.append(eval_result)

        success_batch_groups = self.divide_list(len(success_dataset), 4)
        failure_batch_groups = self.divide_list(len(failure_dataset), 4)

        # if self.target_subgroup in ["success", "all"]:
        #     success_groups = self.create_embedding_groups(success_dataset, embedding_map)
        #     print(success_groups)
        #     for group in success_groups:
        #         if len(group) <= 4:
        #             success_batch_groups.append(group)
        #         else:
        #             for i in range(0, len(group), 4):
        #                 success_batch_groups.append(group[i:i+4])

        # if self.target_subgroup in ["failure", "all"]:
        #     failure_groups = self.create_embedding_groups(failure_dataset, embedding_map)
        #     print(failure_groups)
        #     for group in failure_groups:
        #         if len(group) <= 4:
        #             failure_batch_groups.append(group)
        #         else:
        #             for i in range(0, len(group), 4):
        #                 failure_batch_groups.append(group[i:i+4])

        print(f"Success Batch Groups : {len(success_batch_groups)}")
        print(f"Failure Batch Groups : {len(failure_batch_groups)}")

        best_prompt = prompt
        global_step = 0
        _, _, trainset_global_result = await self._evaluate(trainset, best_prompt)
        trainset_best_score = trainset_global_result.score

        if self.target_subgroup in ["success", "all"]:
            for group in success_batch_groups:
                feedback_history = []
                prompt_history = []
                print(f"Step : {global_step}")
                retry_count = 0
                global_step += 1

                score_report = {}
                while retry_count < self.max_proposals_per_step:
                    retry_count += 1
                    feedback = await self.generate_feedback(
                        prompt=best_prompt,
                        data_indices=group,
                        trainset=success_dataset,
                        predictions=success_predictions,
                        eval_results=success_eval_results,
                        type="success",
                        feedback_history=feedback_history,
                    )
                    new_prompt = await self.apply_feedback(
                        prompt=best_prompt, feedback=feedback, prompt_history=prompt_history
                    )

                    group_trainset = [success_dataset[i] for i in group]
                    _, _, group_trainset_global_result = await self._evaluate(group_trainset, new_prompt)
                    if group_trainset_global_result.score != 1.0:
                        print(
                            f"Trial {retry_count} failed in batch : 1.0 -> {group_trainset_global_result.score}"
                        )
                        score_report = {"step": global_step, "score": 0.0}
                        feedback_history.append({"feedback": feedback, "score": 0.0})
                        prompt_history.append({"prompt": new_prompt, "score": 0.0})
                        continue
                    # validate on trainset
                    _, _, trainset_global_result = await self._evaluate(trainset, new_prompt)
                    if trainset_global_result.score == 1.0:
                        print(f"Trial {retry_count} succeeded in batch, 1.0")
                        score_report = {"step": global_step, "score": trainset_global_result.score}
                        report.feedbacks.append({"type": "success group", "feedback": feedback})
                        report.scores.append(score_report)
                        return new_prompt, report

                    score_report = {"step": global_step, "score": trainset_global_result.score}
                    if trainset_global_result.score > trainset_best_score:
                        print(
                            f"Trial {retry_count} success, {trainset_best_score} -> {trainset_global_result.score}"
                        )
                        best_prompt = new_prompt
                        trainset_best_score = trainset_global_result.score
                        break
                    print(
                        f"Trial {retry_count} failed, {trainset_best_score} -> {trainset_global_result.score}"
                    )

                    feedback_history.append({"feedback": feedback, "score": trainset_global_result.score})
                    prompt_history.append({"prompt": new_prompt, "score": trainset_global_result.score})

                report.scores.append(score_report)
                report.feedbacks.append({"type": "success group", "feedback": feedback})

        if self.target_subgroup in ["failure", "all"]:
            for group in failure_batch_groups:
                retry_count = 0
                global_step += 1
                print(f"Step : {global_step}")
                score_report = {}
                feedback_history = []
                prompt_history = []

                while retry_count < self.max_proposals_per_step:
                    feedback = await self.generate_feedback(
                        prompt=best_prompt,
                        data_indices=group,
                        trainset=failure_dataset,
                        predictions=failure_predictions,
                        eval_results=failure_eval_results,
                        type="failure",
                        feedback_history=feedback_history,
                    )

                    retry_count += 1
                    new_prompt = await self.apply_feedback(
                        prompt=best_prompt, feedback=feedback, prompt_history=prompt_history
                    )
                    new_prompt_messages = [json.dumps(message) for message in new_prompt.messages]
                    new_prompt_messages_str = "\n".join(new_prompt_messages)

                    # validate on trainset batch
                    group_trainset = [failure_dataset[i] for i in group]
                    _, _, group_trainset_global_result = await self._evaluate(group_trainset, new_prompt)
                    if group_trainset_global_result.score == 0.0:
                        score_report = {"step": global_step, "score": 0.0}
                        print(f"Trial {retry_count} failed in batch : 0.0 -> 0.0")
                        feedback_history.append({"feedback": feedback, "score": 0.0})
                        prompt_history.append({"prompt": new_prompt, "score": 0.0})
                        continue        
                    # validate on trainset
                    _, _, trainset_global_result = await self._evaluate(trainset, new_prompt)
                    if trainset_global_result.score == 1.0:
                        print(f"Trial {retry_count} succeeded in batch, 1.0")
                        score_report = {"step": global_step, "score": trainset_global_result.score}
                        report.feedbacks.append({"type": "failure group", "feedback": feedback})
                        report.scores.append(score_report)
                        return new_prompt, report

                    score_report = {"step": global_step, "score": trainset_global_result.score}
                    if trainset_global_result.score > trainset_best_score:
                        print(
                            f"Trial {retry_count} success, {trainset_best_score} -> {trainset_global_result.score}"
                        )
                        best_prompt = new_prompt
                        trainset_best_score = trainset_global_result.score
                        break

                    print(
                        f"Trial {retry_count} failed, {trainset_best_score} -> {trainset_global_result.score}"
                    )

                    feedback_history.append({"feedback": feedback, "score": trainset_global_result.score})
                    prompt_history.append({"prompt": new_prompt, "score": trainset_global_result.score})

                report.scores.append(score_report)
                report.feedbacks.append({"type": "failure group", "feedback": feedback})
                
        report.best_score = trainset_best_score
        return best_prompt, report

    # async def create_embedding_map(self, dataset: List[DatasetItem]) -> List[Dict[str, List[float]]]:
    #     embedding_map = [{} for _ in range(len(dataset))]
    #     for key in dataset[0].inputs.keys():
    #         texts = [item.inputs[key] for item in dataset]
    #         for i in range(0, len(texts), 96):
    #             batch = texts[i:i+96]
    #             batch_embeddings = await aembedding(
    #                 model="text-embedding-3-small",
    #                 input=batch
    #             )
    #             batch_embeddings = [emb['embedding'] for emb in batch_embeddings.data]
    #             for j, emb in enumerate(batch_embeddings):
    #                 embedding_map[i+j][key] = emb
    #     return embedding_map

    # def create_embedding_groups(self, dataset: List[DatasetItem], embedding_map: List[Dict[str, List[float]]]) -> List[List[int]]:
    #     groups = []
    #     for key in dataset[0].inputs.keys():
    #         groups.append(self.group_by_similarity(dataset, embedding_map, key))
    #     # make group as list
    #     # Remove duplicates while preserving order
    #     # delete length 0 list
    #     groups = [group for group in groups if len(group) > 0]
    #     unique_groups = []
    #     for group in groups:
    #         if group not in unique_groups:
    #             unique_groups.append(group)
    #     return unique_groups

    # def group_by_similarity(
    #     self,
    #     dataset: List[DatasetItem],
    #     embedding_map: List[Dict[str, List[float]]],
    #     key: str
    # ):
    #     # Convert embeddings to numpy array
    #     embeddings = np.array([embedding_map[i][key] for i in range(len(dataset))], dtype='float32')

    #     # Normalize the embeddings
    #     embeddings_normalized = embeddings / np.linalg.norm(embeddings, axis=1, keepdims=True)

    #     # Compute the cosine similarity matrix
    #     cosine_similarity_matrix = np.dot(embeddings_normalized, embeddings_normalized.T)

    #     similarity_threshold = 0.9
    #     while similarity_threshold >= 0.7:
    #         # Apply the threshold to create adjacency matrix
    #         adjacency_matrix = cosine_similarity_matrix >= similarity_threshold

    #         # Remove self-loops
    #         np.fill_diagonal(adjacency_matrix, False)

    #         # Build graph from adjacency matrix
    #         G = nx.from_numpy_array(adjacency_matrix)

    #         # Find connected components
    #         groups = [list(component) for component in nx.connected_components(G)]
    #         groups = [group for group in groups if len(group) > 1]

    #         if len(groups) > 0:
    #             print("Find groups with threshold : ", similarity_threshold)
    #             return groups
    #         print("No groups found with threshold : ", similarity_threshold)

    #         similarity_threshold -= 0.05

    #     return []

    async def generate_feedback(
        self,
        prompt: Prompt,
        data_indices: List[int],
        trainset: List[DatasetItem],
        predictions: List[Any],
        eval_results: List[MetricResult],
        feedback_history: List[Dict[str, Any]],
        type: Literal["success", "failure"],
    ) -> str:
        if type == "success":
            feedback_generator = self.success_feedback_generator
        else:
            feedback_generator = self.failure_feedback_generator

        prompt_messages = [json.dumps(message) for message in prompt.messages]
        prompt_messages_str = "\n".join(prompt_messages)

        feedback_history_str = ""
        for history in feedback_history:
            feedback_history_str += f"Feedback : {history['feedback']}\n"
            feedback_history_str += f"Score : {history['score']}\n"

        report = ""
        for index in data_indices:
            data = trainset[index]
            pred = predictions[index]
            eval_result = eval_results[index]
            report += f"Input : {str(data['inputs'])}\n"
            if type != "success":
                report += f"Ground Truth : {str(data['outputs'])}\n"
            report += f"Generator Prediction: {str(pred)}\n"
            report += f"Metric Result: {str(eval_result)}\n"

        retry_count = 0
        while retry_count < 3:
            try:
                response = await feedback_generator(
                    task_description=self.task_description,
                    metric_description=self.metric_description,
                    base_prompt=prompt_messages_str,
                    report=report,
                    feedback_history=feedback_history_str,
                )

                if not response.strip().startswith("{"):
                    response = "{" + response

                response_json = json.loads(response)
                if response_json.get("feedback", None) is not None:
                    return response_json["feedback"]
                else:
                    retry_count += 1
            except Exception as e:
                retry_count += 1

        raise Exception("Failed to generate feedback")

    async def apply_feedback(
        self, prompt: Prompt, feedback: str, prompt_history: List[Dict[str, Any]]
    ):
        retry_count = 0
        prompt_messages = [json.dumps(message) for message in prompt.messages]
        prompt_messages_str = "\n".join(prompt_messages)

        prompt_history_str = ""
        for history in prompt_history:
            prompt_history_str += f"Prompt : {json.dumps(history['prompt'].messages)}\n"
            prompt_history_str += f"Score : {history['score']}\n"

        while retry_count < 3:
            try:
                new_prompt_str = await self.feedback_applier(
                    task_description=self.task_description,
                    base_prompt=prompt_messages_str,
                    feedback=feedback,
                    prompt_history=prompt_history_str,
                )
                if not new_prompt_str.strip().startswith("```prompt"):
                    new_prompt_str = "```prompt\n" + new_prompt_str

                extracted_prompt = extract_prompt(new_prompt_str)
                new_prompt_message = Prompt.load(extracted_prompt)
                new_prompt = prompt.deepcopy()
                new_prompt.messages = new_prompt_message.messages

                messages = [json.dumps(message) for message in new_prompt.messages]
                messages_str = "\n".join(messages)
                if (
                    new_prompt.response_format is not None
                    and new_prompt.response_format["type"] == "json_object"
                    and "json" not in messages_str
                ):
                    # add "json" to the messages
                    new_prompt = await reformat_prompt(new_prompt, new_prompt.response_format)

                return new_prompt
            except Exception as e:
                print(e)
                retry_count += 1

        raise Exception("Failed to apply feedback")

    def divide_list(self, list_length: int, group_size: int) -> List[List[int]]:
        # Create a list of integers from 0 to list_length - 1
        full_list = list(range(list_length))
        # Shuffle the list randomly
        random.shuffle(full_list)

        # Divide the list into groups of size 'group_size'
        groups = [full_list[i : i + group_size] for i in range(0, list_length, group_size)]

        # Remove the last group if its size is 1
        if len(groups) > 0 and len(groups[-1]) == 1:
            groups.pop()

        return groups
