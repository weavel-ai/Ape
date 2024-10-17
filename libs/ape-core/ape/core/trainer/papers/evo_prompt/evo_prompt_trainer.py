import asyncio
import copy
import json
import random
import heapq
import numpy as np
from typing import Any, Dict, List, Optional, Tuple, Literal
from ape.common.prompt import Prompt
from ape.common.types import GlobalMetricResult, MetricResult, DatasetItem
from ape.common.generator import BaseGenerator
from ape.common.global_metric import BaseGlobalMetric
from ape.common.metric import BaseMetric
from ape.common.utils.logging import logger
from ape.core.utils import extract_prompt, reformat_prompt
from ape.core.core_prompts import ApeCorePrompts
from ape.core.trainer.base import BaseTrainer
from ape.core.types.report import EvoPromptReport

class EvoPromptTrainer(BaseTrainer):
    def __init__(
        self,
        generator: BaseGenerator,
        metric: BaseMetric,
        global_metric: BaseGlobalMetric,
        evolution_method: Literal['para', 'ga', 'de'] = 'para',
        popsize: int = 10,
        epoch: int = 10,
        parent_selection_mode: Literal['random', 'wheel', 'tour'] = 'wheel',
        child_selection_mode: Literal['child', 'topk'] = 'topk',
        **kwargs,
    ):
        super().__init__(generator, metric, global_metric, **kwargs)
        self.evolution_method = evolution_method
        self.popsize = popsize
        self.epoch = epoch
        self.population: List[int] = []
        self.new_children: List[int] = []
        self.indices2prompts: Dict[int, Prompt] = {}
        self.evaluated_prompts: Dict[int, float] = {}
        self.prompts2mark: Dict[int, str] = {}
        self.cur_epoch = -1
        self.parent_selection_mode = parent_selection_mode
        self.child_selection_mode = child_selection_mode
        self.next_index = 0  # To generate unique indices

        # Load the evolution prompt template
        self.evolution_prompt_de = ApeCorePrompts.get("evolution-prompt-de")
        self.evolution_prompt_ga = ApeCorePrompts.get("evolution-prompt-ga")
        # Load the paraphraser prompt
        self.paraphraser_prompt = ApeCorePrompts.get("evolution-prompt-para")

        self.lock = asyncio.Lock()

    async def _add_prompt(self, prompt: Prompt) -> int:
        async with self.lock:
            index = self.next_index
            self.indices2prompts[index] = prompt
            self.next_index += 1
            return index

    async def train(
        self,
        prompt: Prompt,
        trainset: List[DatasetItem],
        valset: List[DatasetItem],
    ) -> Tuple[Prompt, EvoPromptReport]:
        # Initialize population
        report = EvoPromptReport(scores=[], best_score=0.0)
        print("Initializing population...")
        await self.init_pop(prompt, trainset, valset, report)
        print("Population initialized")
        
        best_scores = []
        avg_scores = []

        # Evolution loop
        for step in range(self.cur_epoch + 1, self.epoch):
            logger.info(f"Step: {step}")
            print(f"Step: {step}")
            # Generate new prompts
            print("Generating new prompts...")
            await self.generate_new_prompts(trainset)
            print("New prompts generated")

            # Evaluate the new population
            print("Evaluating population...")
            await self.evaluate_population(trainset)
            print("Population evaluated")
            
            # Record best and average scores
            total_score = sum(self.evaluated_prompts[p] for p in self.population)
            best_score = max(self.evaluated_prompts[p] for p in self.population)
            avg_score = total_score / len(self.population)
            best_scores.append(best_score)
            avg_scores.append(avg_score)

            # Optionally write step results
            logger.info(f"Step {step}: Best Score = {best_score}, Avg Score = {avg_score}")
            print(f"Step {step}: Best Score = {best_score}, Avg Score = {avg_score}")
            
            if self.testmode:
                semaphore = asyncio.Semaphore(5)
                async def eval_with_semaphore(p):
                    async with semaphore:
                        return await self._evaluate(valset, self.indices2prompts[p])
                val_eval_tasks = [eval_with_semaphore(p) for p in self.population]
                val_results = await asyncio.gather(*val_eval_tasks)
                val_scores = [global_score.score for _, _, global_score in val_results]
                best_val_score = max(val_scores)
                
                best_score_prompt_index = max(self.evaluated_prompts.items(), key=lambda x: x[1])[0]
                best_score_prompt_population_index = self.population.index(best_score_prompt_index)
                best_score_prompt_val_score = val_scores[best_score_prompt_population_index]
                
                report.scores.append({
                    "step": step,
                    "best_score": best_score,
                    "avg_score": avg_score,
                    "val_best_score": best_val_score,
                    "val_score": best_score_prompt_val_score,
                    "val_avg_score": sum(val_scores) / len(val_scores)
                })
            else:
                report.scores.append({
                    "step": step,
                    "best_score": best_score,
                    "avg_score": avg_score
                })
            
            if best_score >= 1.0:
                break

        # After evolution, select the best prompt
        best_prompt_index = max(self.evaluated_prompts.items(), key=lambda x: x[1])[0]
        best_prompt = self.indices2prompts[best_prompt_index]
        return best_prompt, report

    async def init_pop(self, prompt: Prompt, trainset: List[DatasetItem], valset: List[DatasetItem], report: EvoPromptReport):
        # Function to paraphrase a single prompt
        async def _paraphrase_prompt(prompt: Prompt) -> Prompt:
            # Use the paraphraser prompt to paraphrase the prompt
            prompt_messages = [json.dumps(message) for message in prompt.messages]
            prompt_messages_str = "\n".join(prompt_messages)

            # Generate paraphrased prompt
            paraphrased_prompt_raw = await self.paraphraser_prompt(base_prompt=prompt_messages_str)
            # Ensure the paraphrased prompt starts with the expected format
            if not paraphrased_prompt_raw.strip().startswith("```prompt"):
                paraphrased_prompt_raw = "```prompt\n" + paraphrased_prompt_raw
                
            paraphrased_prompt_messages_str = extract_prompt(paraphrased_prompt_raw)
            # Load into Prompt object
            new_prompt_message = Prompt.load(paraphrased_prompt_messages_str)
            if not new_prompt_message.messages:
                logger.warning("Generated prompt has no messages")
                raise ValueError("Generated prompt has no messages")
            paraphrased_prompt = prompt.deepcopy()
            paraphrased_prompt.messages = new_prompt_message.messages
            return paraphrased_prompt

        # Create paraphrased prompts in parallel
        paraphrase_tasks = [ _paraphrase_prompt(prompt) for _ in range(self.popsize) ]
        paraphrased_prompts = await asyncio.gather(*paraphrase_tasks)

        # Add paraphrased prompts to the population
        self.population = [await self._add_prompt(p) for p in paraphrased_prompts]
        self.prompts2mark = {p: "paraphrased_initial" for p in self.population}

        # Evaluate the initial population in parallel using a semaphore
        semaphore = asyncio.Semaphore(5)  # Adjust the value as needed
        async def evaluate_with_semaphore(p):
            async with semaphore:
                return await self._evaluate(trainset, self.indices2prompts[p])
        
        eval_tasks = [
            evaluate_with_semaphore(p) for p in self.population
        ]
        eval_results = await asyncio.gather(*eval_tasks)
        for p, (_, _, global_score) in zip(self.population, eval_results):
            self.evaluated_prompts[p] = global_score.score
        
        if self.testmode:
            semaphore = asyncio.Semaphore(5)  # Limit concurrent evaluations to 5
            async def evaluate_with_semaphore(p):
                async with semaphore:
                    return await self._evaluate(valset, self.indices2prompts[p])
            
            val_eval_tasks = [
                evaluate_with_semaphore(p) for p in self.population
            ]
            val_results = await asyncio.gather(*val_eval_tasks)
            val_scores = [global_score.score for _, _, global_score in val_results]
            best_val_score = max(val_scores)
            
            best_score_prompt_index = max(self.evaluated_prompts.items(), key=lambda x: x[1])[0]
            best_score_prompt_population_index = self.population.index(best_score_prompt_index)
            best_score_prompt_val_score = val_scores[best_score_prompt_population_index]
            
            report.scores.append({
                "step": -1,
                "best_score": max(self.evaluated_prompts.values()),
                "avg_score": sum(self.evaluated_prompts.values()) / len(self.evaluated_prompts),
                "val_best_score": best_val_score,
                "val_score": best_score_prompt_val_score,
                "val_avg_score": sum(val_scores) / len(val_scores)
            })
        else:
            report.scores.append({
                "step": -1,
                "best_score": max(self.evaluated_prompts.values()),
                "avg_score": sum(self.evaluated_prompts.values()) / len(self.evaluated_prompts)
            })
        report.best_score = max(self.evaluated_prompts.values())

    async def evaluate_population(self, trainset: List[DatasetItem]):
        # Evaluate each prompt in the population using a semaphore
        semaphore = asyncio.Semaphore(5)  # Limit concurrent evaluations to 5
        eval_tasks = []

        async def evaluate_with_semaphore(prompt_index):
            async with semaphore:
                return prompt_index, await self._evaluate(trainset, self.indices2prompts[prompt_index])

        for prompt_index in self.population:
            if prompt_index not in self.evaluated_prompts:
                eval_tasks.append(evaluate_with_semaphore(prompt_index))

        results = await asyncio.gather(*eval_tasks) if eval_tasks else []
        for (prompt_index, _), (_, _, global_score) in zip(eval_tasks, results):
            self.evaluated_prompts[prompt_index] = global_score.score

    async def generate_new_prompts(self, trainset: List[DatasetItem]):
        # Call the appropriate method based on evolution_method
        print(f"Generating new prompts with {self.evolution_method}...")
        if self.evolution_method == 'para':
            await self.generate_new_prompts_para(trainset)
        elif self.evolution_method == 'ga':
            await self.generate_new_prompts_ga(trainset)
        elif self.evolution_method == 'de':
            await self.generate_new_prompts_de(trainset)
        else:
            raise ValueError(f"Unknown evolution method: {self.evolution_method}")
        print("New prompts generated")
        print("Start Selecting Next Generation...")
        # Update the population based on child_selection_mode
        if self.child_selection_mode == 'child':
            # Completely replace the population with new children
            new_population = self.new_children
        elif self.child_selection_mode == 'topk':
            # Select the top `popsize` prompts based on evaluated scores
            sorted_prompts = sorted(
                self.evaluated_prompts.items(),
                key=lambda item: item[1],
                reverse=True
            )
            new_population = [prompt_index for prompt_index, _ in sorted_prompts[:self.popsize]]
        else:
            raise ValueError(f"Unknown child selection mode: {self.child_selection_mode}")
        print("Next Generation Selected")
        self.population = new_population

    async def generate_new_prompts_ga(self, trainset: List[DatasetItem]):
        k = self.popsize
        fitness = np.array([self.evaluated_prompts[prompt_index] for prompt_index in self.population])

        if self.parent_selection_mode == "wheel":
            probabilities = fitness / fitness.sum()
            parent_indices = np.random.choice(
                np.arange(k),
                size=k * 2,  # Each child needs two parents
                replace=True,
                p=probabilities,
            )
            parent_pairs = [(parent_indices[i], parent_indices[i + 1]) for i in range(0, k * 2, 2)]
            parent_pop = [self.population[i] for i in parent_indices]
        elif self.parent_selection_mode in ["random", "tour"]:
            parent_pairs = [random.sample(self.population, 2) for _ in range(k)]
        else:
            raise ValueError(f"Invalid selection mode: {self.parent_selection_mode}")

        async def create_child(cand_a: int, cand_b: int) -> int:
            # Use the evolution prompt to generate a new prompt
            child_prompt_raw = await self.evolution_prompt_ga(
                prompt1=str(self.indices2prompts[cand_a]),
                prompt2=str(self.indices2prompts[cand_b])
            )
            child_prompt_raw = child_prompt_raw["mutation_prompt"]
            child_prompt_messages_str = extract_prompt(child_prompt_raw)
            # Load into Prompt object
            new_prompt_message = Prompt.load(child_prompt_messages_str)
            if not new_prompt_message.messages:
                logger.warning(f"Generated prompt has no messages, {child_prompt_raw}")
                raise ValueError(f"Generated prompt has no messages, {child_prompt_raw}")
            child_prompt = self.indices2prompts[cand_a].deepcopy()  # Use cand_a as base
            child_prompt.messages = new_prompt_message.messages
            child_prompt_index = await self._add_prompt(child_prompt)
            self.prompts2mark[child_prompt_index] = "evolved"
            
            # Evaluate the new prompt if not already evaluated
            if child_prompt_index not in self.evaluated_prompts:
                _, _, global_score = await self._evaluate(trainset, child_prompt)
                self.evaluated_prompts[child_prompt_index] = global_score.score
            
            return child_prompt_index

        # Create a list of tasks for parallel execution
        tasks = [create_child(cand_a, cand_b) for cand_a, cand_b in parent_pairs]
        new_children = await asyncio.gather(*tasks)

        self.new_children = new_children

    async def generate_new_prompts_de(self, trainset: List[DatasetItem]):
        k = self.popsize

        async def create_de_child(j: int) -> int:
            old_prompt_index = self.population[j]

            # Select candidates
            candidates = [self.population[i] for i in range(k) if i != j]
            if len(candidates) < 3:
                candidates = self.population.copy()
            a, b, c = random.sample(candidates, 3)

            # Use the evolution prompt to generate a new prompt
            new_prompt_raw = await self.evolution_prompt_de(
                base_prompt=str(self.indices2prompts[old_prompt_index]),
                prompt1=str(self.indices2prompts[a]),
                prompt2=str(self.indices2prompts[b]),
                prompt3=str(self.indices2prompts[c])
            )
            new_prompt_raw = new_prompt_raw["final_prompt"]
            new_prompt_messages_str = extract_prompt(new_prompt_raw)
            # Load into Prompt object
            new_prompt_message = Prompt.load(new_prompt_messages_str)
            if not new_prompt_message.messages:
                logger.warning("Generated prompt has no messages")
                raise ValueError("Generated prompt has no messages")
            de_prompt = self.indices2prompts[old_prompt_index].deepcopy()
            de_prompt.messages = new_prompt_message.messages

            de_prompt_index = await self._add_prompt(de_prompt)
            self.prompts2mark[de_prompt_index] = "evolved"

            # Evaluate the new prompt if not already evaluated
            if de_prompt_index not in self.evaluated_prompts:
                _, _, global_score = await self._evaluate(trainset, de_prompt)
                self.evaluated_prompts[de_prompt_index] = global_score.score

            return de_prompt_index

        # Create a list of tasks for parallel execution
        tasks = [create_de_child(j) for j in range(k)]
        new_children = await asyncio.gather(*tasks)

        self.new_children = new_children

    async def generate_new_prompts_para(self, trainset: List[DatasetItem]):
        # For paraphrasing, we paraphrase the top k prompts
        k = self.popsize

        # If we haven't initialized the heap, do so
        if not hasattr(self, 'topk_heap'):
            # Initialize the heap with initial prompts
            self.topk_heap = []
            for prompt_index in self.population:
                score = self.evaluated_prompts.get(prompt_index, 0)
                heapq.heappush(self.topk_heap, (score, prompt_index))

        # Get the top k prompts
        top_k = heapq.nlargest(k, self.topk_heap)
        top_k_prompt_indices = [item[1] for item in top_k]

        async def _paraphrase_prompt(prompt: Prompt) -> Prompt:
            # Use the paraphraser prompt to paraphrase the prompt
            prompt_messages = [json.dumps(message) for message in prompt.messages]
            prompt_messages_str = "\n".join(prompt_messages)

            # Generate paraphrased prompt
            paraphrased_prompt_raw = await self.paraphraser_prompt(base_prompt=prompt_messages_str)
            # Extract the prompt
            if not paraphrased_prompt_raw.strip().startswith("```prompt"):
                paraphrased_prompt_raw = "```prompt\n" + paraphrased_prompt_raw
                
            paraphrased_prompt_messages_str = extract_prompt(paraphrased_prompt_raw)
            # Load into Prompt object
            new_prompt_message = Prompt.load(paraphrased_prompt_messages_str)
            if not new_prompt_message.messages:
                logger.warning("Generated prompt has no messages")
                raise ValueError("Generated prompt has no messages")
            paraphrased_prompt = prompt.deepcopy()
            paraphrased_prompt.messages = new_prompt_message.messages
            return paraphrased_prompt
        
        # Paraphrase the top k prompts
        paraphrased_prompts = await asyncio.gather(
            *[_paraphrase_prompt(self.indices2prompts[prompt_index]) for prompt_index in top_k_prompt_indices]
        )

        new_children = []
        # Evaluate the paraphrased prompts
        for p in paraphrased_prompts:
            p_index = await self._add_prompt(p)
            self.prompts2mark[p_index] = "paraphrased"
            if p_index not in self.evaluated_prompts:
                _, _, global_score = await self._evaluate(trainset, p)
                self.evaluated_prompts[p_index] = global_score.score
            new_children.append(p_index)
            heapq.heappush(self.topk_heap, (self.evaluated_prompts[p_index], p_index))

        self.new_children = new_children