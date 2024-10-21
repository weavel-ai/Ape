import asyncio
import copy
import json
import random
import numpy as np
from tqdm import tqdm
from typing import Any, Dict, List, Literal, Optional, Tuple
from ape.common.prompt.prompt_base import Prompt
from ape.common.types import GlobalMetricResult, MetricResult, DatasetItem
from ape.common.generator import BaseGenerator
from ape.common.global_metric import BaseGlobalMetric
from ape.common.metric import BaseMetric
from ape.common.utils.logging import logger
from ape.core.utils import reformat_prompt
from ape.core.core_prompts import ApeCorePrompts
from ape.core.trainer.base import BaseTrainer
from ape.core.types.report import TextGradEvoTrainerReport


class TextGradEvoTrainer(BaseTrainer):
    def __init__(
        self,
        generator: BaseGenerator,
        metric: BaseMetric,
        global_metric: BaseGlobalMetric,
        random_seed: int = 42,
        max_evolution_per_step: int = 5,
        population_size: int = 5,
        evolution_method: Literal["para", "ga", "de", None] = "ga",
        parent_selection_mode: Literal['random', 'wheel', 'tour'] = 'wheel',
        child_selection_mode: Literal['child', 'topk'] = 'topk',
        **kwargs,
    ):
        super().__init__(generator, metric, global_metric, **kwargs)
        self.random_seed = random_seed
        
        self.max_evolution_per_step = max_evolution_per_step
        self.population_size = population_size
        self.evolution_method = evolution_method
        self.parent_selection_mode = parent_selection_mode
        self.child_selection_mode = child_selection_mode

        self.text_gradient_generator_prompt = ApeCorePrompts.get("text-grad-evo-generator")
        self.text_gradient_applier_prompt = ApeCorePrompts.get("text-grad-evo-applier")
        
        # Load the evolution prompt template
        self.evolution_prompt_de = ApeCorePrompts.get("evoprompt-prompt-de")
        self.evolution_prompt_ga = ApeCorePrompts.get("evoprompt-prompt-ga")
        self.paraphraser_prompt = ApeCorePrompts.get("evoprompt-prompt-para")

        random.seed(random_seed)
        np.random.seed(random_seed)

    async def train(
        self,
        prompt: Prompt,
        trainset: List[DatasetItem],
        valset: List[DatasetItem],
    ) -> Tuple[Prompt, TextGradEvoTrainerReport]:
        """
        Train the model using text gradients.

        Args:
            prompt (Prompt): The initial prompt to start training.
            trainset (List[DatasetItem]): The training dataset.
            valset (List[DatasetItem]): The validation dataset.

        Returns:
            Tuple[Prompt, Optional[TextGradientTrainerReport]]: The best prompt and an optional training report.
        """

        # Initialize Text Gradient Trainer Report
        report = TextGradEvoTrainerReport(scores=[], evolution_steps=[], best_score=0.0, trials=0, success=0)

        # Step 1: Shuffle the training set without modifying the original list
        shuffled_trainset = copy.deepcopy(trainset)
        random.shuffle(shuffled_trainset)

        # Step 2: Initialize best_prompt and failed_count
        best_prompt = prompt

        _, best_trainset_results, best_trainset_global_result = await self._evaluate(
            dataset=trainset,
            prompt=best_prompt
        )
        best_trainset_score = best_trainset_global_result.score

        # Iterate over the shuffled training set in batches
        for i in tqdm(
            range(0, len(shuffled_trainset)), desc="Training Dataset"
        ):
            target_example = shuffled_trainset[i]
            target_pred, target_eval_result, target_global_result = await self._evaluate(
                dataset=[target_example],
                prompt=best_prompt
            )
            target_score = target_global_result.score

            if target_score == 1.0:
                logger.debug("Target score is 1.0, skipping text gradient generation")
                if self.testmode:
                    _, _, val_global_result  = await self._evaluate(valset, best_prompt)
                    report.scores.append({"step": i, "score": best_trainset_score, "val_score": val_global_result.score})
                else:
                    report.scores.append({"step": i, "score": best_trainset_score})
                continue

            text_gradients = await asyncio.gather(  
                *[
                    self._text_gradient_generator(
                        prompt=best_prompt,
                        inputs=target_example["inputs"],
                        outputs=target_example["outputs"],
                        generator_output=target_pred,
                        metric_result=target_eval_result,
                        parallel_task_id=j
                    )
                    for j in range(self.population_size)
                ]
            )  # TODO: fix the score threshold into dynamic)
            text_gradients = [tg for tg in text_gradients if tg]  # Filter out empty gradients
            
            # TODO: start evolution loop here
            
            parent_generation_prompts = await asyncio.gather(
                *[
                    self._text_gradient_applier(
                        prompt=best_prompt,
                        text_gradient="\n".join(text_gradient),
                    )
                    for text_gradient in text_gradients
                ]   
            )
            parent_generation_scores = await asyncio.gather(
                *[
                    self._evaluate(dataset=trainset, prompt=prompt)
                    for prompt in parent_generation_prompts
                ]
            )
            parent_generation_scores = [score[2].score for score in parent_generation_scores]
            
            best_score = max(parent_generation_scores)
            best_score_index = parent_generation_scores.index(best_score)
            if best_score > best_trainset_score:
                best_prompt = parent_generation_prompts[best_score_index]
                best_trainset_score = best_score
                if self.testmode:
                    _, _, val_global_result  = await self._evaluate(valset, best_prompt)
                    report.scores.append({"step": len(report.scores), "score": best_trainset_score, "val_score": val_global_result.score})
                else:
                    report.scores.append({"step": len(report.scores), "score": best_trainset_score})
                
                evolution_step_info = {
                    "best_score": best_trainset_score,
                    "average_score": sum(parent_generation_scores) / len(parent_generation_scores),
                }
                if self.testmode:
                    _, _, val_global_result  = await self._evaluate(valset, best_prompt)
                    evolution_step_info["val_score"] = val_global_result.score
                report.evolution_steps.append(evolution_step_info)
            else:
                if self.evolution_method is None:
                    report.best_score = best_trainset_score
                    if self.testmode:
                        _, _, val_global_result  = await self._evaluate(valset, best_prompt)
                        report.scores.append({"step": i, "score": best_trainset_score, "val_score": val_global_result.score})
                    else:
                        report.scores.append({"step": i, "score": best_trainset_score})
                else:
                    logger.debug("Start evolution loop")
                    evolution_step = 0
                    while evolution_step < self.max_evolution_per_step:
                        evolution_step += 1

                        # Generate new prompts
                        new_generation_prompts = await self.generate_new_generations(
                            parent_generation_prompts,
                            parent_generation_scores
                        )
                        new_generation_scores = await asyncio.gather(
                            *[
                                self._evaluate(dataset=trainset, prompt=prompt)
                                for prompt in new_generation_prompts
                            ]
                        )
                        new_generation_scores = [score[2].score for score in new_generation_scores]
                        
                        best_score = max(new_generation_scores)
                        if best_score > best_trainset_score:
                            best_prompt = new_generation_prompts[new_generation_scores.index(best_score)]
                            best_trainset_score = best_score
                            report.scores.append({"step": i, "score": best_trainset_score})
                            evolution_step_info = {
                                "best_score": best_trainset_score,
                                "average_score": sum(new_generation_scores) / len(new_generation_scores),
                            }
                            if self.testmode:
                                _, _, val_global_result  = await self._evaluate(valset, best_prompt)
                                report.scores.append({"step": i, "score": best_trainset_score, "val_score": val_global_result.score})
                                evolution_step_info["val_score"] = val_global_result.score
                            else:
                                report.scores.append({"step": i, "score": best_trainset_score})    
                            report.evolution_steps.append(evolution_step_info)
                            break
                        
                        else:
                            logger.debug("No improvement in this generation")
                            evolution_step_info = {
                                "best_score": best_trainset_score,
                                "average_score": sum(new_generation_scores) / len(new_generation_scores),
                            }
                            if self.testmode:
                                _, _, val_global_result  = await self._evaluate(valset, best_prompt)
                                evolution_step_info["val_score"] = val_global_result.score
                            report.evolution_steps.append(evolution_step_info)
                            
                            # update generation
                            # best top self.population_size prompts
                            if self.child_selection_mode == 'topk':
                                whole_generations = parent_generation_prompts + new_generation_prompts
                                whole_generation_scores = parent_generation_scores + new_generation_scores
                                # Pair scores with prompts
                                score_prompt_pairs = list(zip(whole_generation_scores, whole_generations))
                                # Sort based on scores
                                sorted_pairs = sorted(score_prompt_pairs, key=lambda x: x[0], reverse=True)
                                # Select top K prompts
                                parent_generation_prompts = [pair[1] for pair in sorted_pairs[:self.population_size]]
                                parent_generation_scores = [pair[0] for pair in sorted_pairs[:self.population_size]]
                            elif self.child_selection_mode == 'child':
                                parent_generation_prompts = new_generation_prompts
                                parent_generation_scores = new_generation_scores
                            else:
                                raise ValueError(f"Unknown child selection mode: {self.child_selection_mode}")
                            
                    logger.debug("Evolution loop ends, prompt improve failed")
                    report.best_score = best_trainset_score
                    if self.testmode:
                        _, _, val_global_result  = await self._evaluate(valset, best_prompt)
                        report.scores.append({"step": i, "score": best_trainset_score, "val_score": val_global_result.score})
                    else:
                        report.scores.append({"step": i, "score": best_trainset_score})
                
        return best_prompt, report

    async def _text_gradient_generator(
        self,
        prompt: Prompt,
        inputs: Dict[str, Any],
        outputs: Any,
        generator_output: Any,
        metric_result: MetricResult,
        parallel_task_id: int
    ) -> Any:
        """
        Generate text gradient based on inputs, outputs, generator outputs, and metric results.
        """
        # TODO: apply EXPEL-style text gradient generator in this function
        retry_count = 0
        
        while retry_count < 3:
            try:
                text_gradient = await self.text_gradient_generator_prompt(
                    task_description=self.task_description,
                    metric_description=self.metric_description,
                    base_prompt=str(prompt.messages),
                    inputs=str(inputs),
                    outputs=str(outputs),
                    generator_output=str(generator_output),
                    metric_result=str(metric_result),
                    _retry_count=retry_count,
                    parallel_task_id=parallel_task_id
                )
                text_gradient = text_gradient.strip()
                if not text_gradient.startswith("{"):
                    text_gradient = "{" + text_gradient

                text_gradient = json.loads(text_gradient)["feedback"]
                return text_gradient
            except Exception as e:
                logger.warning(f"Error generating text gradient: {e}")
                retry_count += 1
                if retry_count >= 3:
                    return ""

    async def _text_gradient_applier(
        self, prompt: Prompt, text_gradient: str
    ) -> Prompt:
        """
        Apply text gradient to the prompt to obtain a new prompt.
        """
        retry_count = 0

        while retry_count < 3:
            try:
                new_prompt_raw = await self.text_gradient_applier_prompt(
                    task_description=self.task_description,
                    base_prompt=str(prompt.messages),
                    feedback=text_gradient,
                    _retry_count=retry_count
                )

                new_prompt_message = new_prompt_raw["messages"]
                new_prompt = prompt.deepcopy()
                new_prompt.messages = new_prompt_message

                # add "json" if response_format is not None & response_format.type is "json_object" & "json" not in messages
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
                logger.warning(f"Error applying text gradient: {e}")
                retry_count += 1

    async def generate_new_generations(
        self,
        parent_generation_prompts: List[Prompt],
        parent_generation_scores: List[float],
    ) -> List[Prompt]:
        # Call the appropriate method based on evolution_method
        if self.evolution_method == 'para':
            return await self.generate_new_generations_para(parent_generation_prompts, parent_generation_scores)
        elif self.evolution_method == 'ga':
            return await self.generate_new_generations_ga(parent_generation_prompts, parent_generation_scores)
        elif self.evolution_method == 'de':
            return await self.generate_new_generations_de(parent_generation_prompts, parent_generation_scores)
        else:
            raise ValueError(f"Unknown evolution method: {self.evolution_method}")


    async def generate_new_generations_ga(
        self, 
        parent_generation_prompts: List[Prompt], 
        parent_generation_scores: List[float]
    ):
        k = self.population_size
        fitness = np.array(parent_generation_scores)

        if self.parent_selection_mode == "wheel":
            probabilities = fitness / fitness.sum()
            parent_indices = np.random.choice(
                np.arange(k),
                size=k * 2,  # Each child needs two parents
                replace=True,
                p=probabilities,
            )
            parent_pairs: List[List[Prompt]] = [
                [
                    parent_generation_prompts[parent_indices[i]],
                    parent_generation_prompts[parent_indices[i + 1]]
                ]
                for i in range(0, k * 2, 2)
            ]
        elif self.parent_selection_mode in ["random", "tour"]:
            parent_pairs: List[List[Prompt]] = [random.sample(parent_generation_prompts, 2) for _ in range(k)]
        else:
            raise ValueError(f"Invalid selection mode: {self.parent_selection_mode}")

        async def create_child(cand_a: Prompt, cand_b: Prompt) -> Prompt:
            # Use the evolution prompt to generate a new prompt
            child_prompt_raw = await self.evolution_prompt_ga(
                prompt1=str(cand_a.messages),
                prompt2=str(cand_b.messages)
            )
            child_prompt_messages = child_prompt_raw["mutation_prompt"]["messages"]
            # Load into Prompt object   
            child_prompt = cand_a.deepcopy()  # Use cand_a as base
            child_prompt.messages = child_prompt_messages
            
            return child_prompt

        # Create a list of tasks for parallel execution
        tasks = [create_child(cand_a, cand_b) for cand_a, cand_b in parent_pairs]
        new_children = await asyncio.gather(*tasks)

        return new_children

    async def generate_new_generations_de(
        self, 
        parent_generation_prompts: List[Prompt], 
        parent_generation_scores: List[float]
    ):
        k = self.population_size

        async def create_de_child(j: int) -> Prompt:
            old_prompt = parent_generation_prompts[j]

            # Select candidates
            candidates = [parent_generation_prompts[i] for i in range(k) if i != j]
            if len(candidates) < 3:
                candidates = parent_generation_prompts.copy()
            a, b, c = random.sample(candidates, 3)

            # Use the evolution prompt to generate a new prompt
            new_prompt_raw = await self.evolution_prompt_de(
                base_prompt=str(old_prompt.messages),
                prompt1=str(a.messages),
                prompt2=str(b.messages),
                prompt3=str(c.messages)
            )
            new_prompt_messages = new_prompt_raw["final_prompt"]["messages"]
            
            de_prompt = old_prompt.deepcopy()
            de_prompt.messages = new_prompt_messages

            return de_prompt

        # Create a list of tasks for parallel execution
        tasks = [create_de_child(j) for j in range(k)]
        new_children = await asyncio.gather(*tasks)

        return new_children

    async def generate_new_generations_para(
        self, 
        parent_generation_prompts: List[Prompt], 
        parent_generation_scores: List[float]
    ):
        async def _paraphrase_prompt(prompt: Prompt) -> Prompt:
            # Use the paraphraser prompt to paraphrase the prompt
            paraphrased_prompt_raw = await self.paraphraser_prompt(base_prompt=str(prompt.messages))
            # Extract the prompt
            paraphrased_prompt_messages = paraphrased_prompt_raw["messages"]
            # Load into Prompt object
            paraphrased_prompt = prompt.deepcopy()
            paraphrased_prompt.messages = paraphrased_prompt_messages
            return paraphrased_prompt
        
        # Paraphrase all prompts from the previous generation
        paraphrased_prompts = await asyncio.gather(
            *[_paraphrase_prompt(prompt) for prompt in parent_generation_prompts]
        )

        return paraphrased_prompts
