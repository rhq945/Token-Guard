# -*- coding: utf-8 -*-
# Phi-Decoding: A decoding algorithm that combines clustering and sampling strategies
# This implementation uses TF-IDF vectorization and K-means clustering for response selection
# Warning: This implementation may be unstable and requires further testing
#
# Parameter Optimization for Better Sampling:
# - num_rollout: 4 -> 10 (增加采样多样性)
# - cluster_num: 3 -> 2 (减少聚类数，避免过度分割)
# - threshold: 0.69 -> 0.75 (提高早停阈值，允许更多步骤)
# - sigma_rate: 1.0 -> 0.8 (降低宽度剪枝强度，保留更多样本)
# - TEMPERATURE: 0.1 -> 0.3 (提高softmax温度，增加选择多样性)
# - sampling temperature: 0.6 -> 0.4 (降低生成温度，提高一致性)
import time
from typing import List, Dict, Any, Optional, Tuple
import numpy as np
import torch
import json
import os
import argparse
from sklearn.metrics import silhouette_score
from sklearn.cluster import KMeans
from sklearn.feature_extraction.text import TfidfVectorizer
from transformers import (
    StoppingCriteriaList,
    StoppingCriteria,
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig
)
from vllm import LLM, SamplingParams
from logic_example import (
    HISTORY_8_FEW_SHOT,
    NFL_8_FEW_SHOT,
    halueval_6_FEW_SHOT,
    covidQA_4_FEW_SHOT,
    financebench_5_FEW_SHOT,
    pubmedQA_4_FEW_SHOT,
    RAGTruth_5_FEW_SHOT
)

# set visible gpus
# os.environ["CUDA_VISIBLE_DEVICES"] = "2,3"


# Constants for algorithm configuration
INF = 10  # Used for initialization of min/max values
TEMPERATURE = 0.3  # Temperature parameter for softmax sampling


def parse_arguments():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description="Phi-Decoding Algorithm")

    # Model configuration
    parser.add_argument('--model_id', type=str, default='llama3.1',
                        help='Model identifier')
    parser.add_argument('--model_path', type=str, default='/data/rhq/TOKEN-GUARD/halu/models/Llama-2-13b-chat',
                        help='Model path')
    parser.add_argument('--gpus', type=int, default=1,
                        help='Number of GPUs to use')

    # Data configuration
    parser.add_argument('--datasets', type=str, default='gsm',
                        help='Dataset type')  # gsm, math, reclor, logiqa, gpqa, arc
    parser.add_argument('--data_path', type=str,
                        default='./data/gsm_test.json',
                        help='Path to input data')
    parser.add_argument('--output_dir', type=str,
                        default='./results/',
                        help='Output directory for results')

    # Algorithm parameters
    parser.add_argument('--step_beam_size', type=int, default=4,
                        help='Beam size for each step')
    parser.add_argument('--num_rollout', type=int, default=10,
                        help='Number of rollouts')
    parser.add_argument('--num_foresight', type=int, default=8,
                        help='Number of foresight steps')
    parser.add_argument('--strategy', type=str, default='cluster',
                        help='Response selection strategy')
    parser.add_argument('--width_pruning_strategy', type=str, default='low_sigma',
                        help='Width pruning strategy')
    parser.add_argument('--depth_pruning_strategy', type=str, default='cluster',
                        help='Depth pruning strategy')
    parser.add_argument('--cluster_num', type=int, default=2,
                        help='Number of clusters for clustering strategy')
    parser.add_argument('--threshold', type=float, default=0.75,
                        help='Threshold for early stopping')
    parser.add_argument('--least_foresight_num', type=int, default=4,
                        help='Minimum number of foresight steps')
    parser.add_argument('--sigma_rate', type=float, default=0.8,
                        help='Sigma rate for width pruning')

    # Execution configuration
    parser.add_argument('--record_process', type=bool, default=True,
                        help='Whether to record the decoding process')
    parser.add_argument('--file_name', type=str, default='test_3',
                        help='Output file name')
    parser.add_argument('--time_path', type=str,
                        default='./results/time/',
                        help='Path to save timing information')
    parser.add_argument('--seed', type=int, default=0,
                        help='Random seed')
    parser.add_argument('--max_examples', type=int, default=50)
    parser.add_argument('--shot_mode', type=str, default='fewshot', choices=['zeroshot', 'fewshot'],
                        help='选择zeroshot或fewshot模式')
    return parser.parse_args()


def softmax(x):
    """
    Compute softmax values for the input array
    Args:
        x: Input array of values
    Returns:
        Softmax probabilities
    """
    e_x = np.exp(np.array(x))
    return e_x / e_x.sum(axis=0)


class PhiDecoder:
    """
    Main class for phi-decoding algorithm implementation.
    Combines clustering and sampling strategies for response selection.
    """

    def __init__(self, args):
        """
        Initialize the decoder
        Args:
            args: Command line arguments containing configuration
        """
        self.args = args
        self.model = None
        self.tokenizer = None
        self.initialize_model()

    def initialize_model(self):
        """Initialize the language model and tokenizer"""   
        model_path = self._get_model_path()
        
        # Load tokenizer without max_length (it's not a valid argument for from_pretrained)
        self.tokenizer = AutoTokenizer.from_pretrained(
            model_path, trust_remote_code=True
        )

        if not self.tokenizer.pad_token:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        # Initialize vLLM engine with a max context length that respects the model's native limit
        self.model = LLM(
            model=model_path,
            tensor_parallel_size=self.args.gpus,
            trust_remote_code=True,
            max_model_len=4096  # Set to 2048 if using original Phi-2; verify via config.json
        )

        np.random.seed(self.args.seed)

    def _get_model_path(self):
        """Get the appropriate model path"""
        return self.args.model_path

    def get_system_prompt(self, dataset_type=None):
        """
        Get the appropriate system prompt based on dataset type
        Args:
            dataset_type: Type of dataset (e.g., 'drop', 'pubmedqa', 'financebench', etc.)
        Returns:
            System prompt string
        """

        # === Define zero-shot prompts for each dataset ===
        zeroshot_map = {
            'pubmedqa': (
                "You will be given a PubMed-style passage and a Yes/No/Maybe question.\n"
                "Answer rules:\n"
                "1. Begin with exactly one of: \"Yes.\" / \"No.\" / \"Maybe.\"\n"
                "2. Summarize the main conclusion from the passage in exactly ONE short sentence (≤25 words).\n"
                "3. Preserve key phrases and medical terms from the passage; do not replace them with synonyms.\n"
                "4. Always include explicitly stated conditions, subgroups, or limitations if they appear in the conclusion.\n"
                "5. Do NOT add recommendations, explanations, or new information.\n"
                "6. The final output must be exactly ONE LINE:\n"
                "Answer:[Yes./No./Maybe. + short sentence]"
            ),
            'financebench': (
                "You are an equity research analyst. Answer the question using **only the data provided**. Follow these instructions carefully:\n\n"
                "1. Always produce a single-line final answer.\n"
                "2. Do not show calculations, reasoning, or commentary.\n"
                "3. Match the exact format of the ground truth:\n"
                "   - \"$360000.00\" for USD thousands\n"
                "   - \"$7223.00\" for USD millions\n"
                "   - \"$4.90\" for USD billions\n"
                "   - \"34.7%\" for percentages\n"
                "   - \"1.08\" for ratios\n"
                "4. If the answer is not directly available from the statements, output:\n"
                "   \"Unable to answer based on given data.\"\n\n"
                "**Example:**\n"
                "Q: How much was Boeing's FY2017 interest expense (USD thousands)?\n"
                "A: Answer: $360000.00\n"
                "At the end, output: Answer:[your answer here]."
            ),
            'halueval': (
                "You will be presented with a question.\n"
                "Answer the user's question strictly based on the given information.\n"
                "Do not make up information.\n"
                "At the end, output: Answer:[your answer here]."
            ),
            'history': (
                "You will be presented with a question.\n"
                "1. Question Analysis: Carefully examine the question to determine the expected answer type:\n"
                "   - 'how many' → number (count)\n"
                "   - 'what year' → specific year from passage\n"
                "   - 'what', 'which', 'who' → name, phrase, or span from passage\n"
                "   - 'when' → time period or date from passage\n"
                "2. Number Handling: Only use numbers that explicitly appear in the passage.\n"
                "   - Do not assume or invent numbers for calculation\n"
                "   - Pay attention to time periods and organizational changes\n"
                "   - For time calculations, use exact years mentioned\n"
                "3. Information Extraction: Extract precise information from the passage.\n"
                "   - Names: Use exact names as they appear\n"
                "   - Numbers: Use exact numbers as stated\n"
                "   - Events: Use specific events mentioned\n"
                "4. Insufficient Information: If the passage does not provide enough information to answer, respond with:\n"
                "   'The passage does not provide enough information to answer this question.'\n"
                "5. Multiple Choice: If the question contains 'or', pay close attention and answer using the options mentioned in the question itself.\n"
                "At the end, output: Answer:[your answer here]."
            ),
            'ragtruth': (
                "You are given passages and a question. Follow these steps:\n"
                "Answer the question using only the information from the given passages.\n"
                " - Include specific examples, numbers, or comparisons if mentioned.\n"
                " - Include all details that support the answer.\n"
                " - Do not add external information.\n"
                " - If the passages do not contain sufficient information, answer: \"Unable to answer based on given passages.\"\n"
                "At the end, output: Answer:[your answer here]"
            ),
            'covidQA': (
                "You will be presented with a question.\n"
                "Answer the user's question strictly based on the given information.\n"
                "Do not make up information.\n"
                "At the end, output: Answer:[your answer here]."
            ),
        }

        default_zeroshot = (
            "You will be presented with a question.\n"
                "Answer the user's question strictly based on the given information.\n"
                "Do not make up information.\n"
                "At the end, output: Answer:[your answer here]."
        )

        fewshot_map = {
            "history": HISTORY_8_FEW_SHOT,
            "nfl": NFL_8_FEW_SHOT,
            "covidQA": covidQA_4_FEW_SHOT,
            "halueval":halueval_6_FEW_SHOT,
            "financebench": financebench_5_FEW_SHOT,
            "pubmedQA": pubmedQA_4_FEW_SHOT,
            "ragtruth": RAGTruth_5_FEW_SHOT,
        }

        if dataset_type is None:
            filename = os.path.basename(self.args.data_path).lower()
            for dtype in zeroshot_map.keys():
                if dtype in filename:
                    dataset_type = dtype
                    break
            else:
                return default_zeroshot

        if self.args.shot_mode == 'zeroshot':
            return zeroshot_map.get(dataset_type, default_zeroshot)
        
        else:  # fewshot 
            base_prompt = zeroshot_map.get(dataset_type)
            if base_prompt is None:
                base_prompt = default_zeroshot

            # 获取 few-shot 示例
            fewshot_examples = fewshot_map.get(dataset_type, "")
            if not fewshot_examples:
                return base_prompt

            system_prompt = (
                f"{base_prompt}\n\n"
                "I will give you some examples for reference:\n"
                f"{fewshot_examples}"
            )
            return system_prompt

    def cluster_responses(self, responses, advantages):
        """
        Cluster responses using TF-IDF and K-means with improved robustness
        Args:
            responses: List of response texts
            advantages: List of advantage values for each response
        Returns:
            Tuple of (clusters, cluster_info)
        """
        # Filter out empty responses
        valid_indices = [i for i, r in enumerate(responses) if r.strip()]
        if len(valid_indices) < self.args.step_beam_size:
            return None, {"state": "cannot cluster", "reason": "insufficient valid responses"}

        try:
            valid_responses = [responses[i] for i in valid_indices]
            
            # Check if responses are too similar for meaningful clustering
            if len(set(valid_responses)) <= 1:
                return None, {"state": "cannot cluster", "reason": "all responses identical"}
            
            # Vectorize responses with optimized preprocessing
            vectorizer = TfidfVectorizer(
                max_features=500,   # Reduced from 1000 to avoid overfitting
                min_df=1,          # Include all terms
                max_df=0.9,        # Slightly adjusted from 0.95
                ngram_range=(1, 1) # Use unigrams only for better stability
            )
            X = vectorizer.fit_transform(valid_responses)
            
            # Check if we have enough features for clustering
            if X.shape[1] < 2:
                return None, {"state": "cannot cluster", "reason": "insufficient features"}
            
            # Perform clustering with silhouette analysis
            kmeans = KMeans(n_clusters=self.args.cluster_num, random_state=42, n_init=10)
            kmeans.fit(X)
            
            # Calculate silhouette score to assess clustering quality
            if len(valid_responses) > 2 and len(set(kmeans.labels_)) > 1:
                silhouette_avg = silhouette_score(X, kmeans.labels_)
                if silhouette_avg < 0.05:  # Relaxed threshold from 0.1 to 0.05
                    return None, {"state": "cannot cluster", "reason": f"poor clustering quality (silhouette: {silhouette_avg:.3f})"}
            else:
                silhouette_avg = 0.0

            # Group responses by cluster
            clusters = [[] for _ in range(self.args.cluster_num)]
            for idx, label in enumerate(kmeans.labels_):
                clusters[label].append(valid_indices[idx])

            return clusters, {
                "state": "success",
                "cluster_sizes": [len(c) for c in clusters],
                "silhouette_score": silhouette_avg,
                "feature_count": X.shape[1]
            }

        except Exception as e:
            return None, {"state": "fail", "error": str(e)}

    def select_response(self, responses, logprobs, advantages):
        """Select final response based on strategy with robustness and semantic bonus"""
        if self.args.strategy == "cluster":
            # filter out empty responses
            valid_indices = [idx for idx, r in enumerate(responses) if r.strip() != '']
            if len(valid_indices) == 0:
                print('all responses in the final generation are empty, use -adv no replace')
                weights = softmax([-adv/TEMPERATURE for adv in advantages])
                return np.random.choice(len(advantages), p=weights)

            if len(valid_indices) < self.args.step_beam_size:
                print('valid responses are less than step_beam_size, use adv no replace')
                weights = softmax([adv/TEMPERATURE for adv in advantages])
                return np.random.choice(len(advantages), p=weights)

            try:
                # prepare cluster data (compress to valid items)
                valid_responses = [responses[i] for i in valid_indices]
                valid_advantages = [advantages[i] for i in valid_indices]

                # Use improved clustering with robustness checks
                clusters, cluster_info = self.cluster_responses(valid_responses, valid_advantages)
                
                if clusters is None:
                    print(f"Clustering failed: {cluster_info.get('reason', 'unknown')}, using advantage-based selection with semantic bonus")
                    # Fallback with semantic quality bonus
                    enhanced_advantages = []
                    for resp, base in zip(valid_responses, valid_advantages):
                        bonus = 0
                        if len(resp.strip()) > 50:
                            bonus += 0.2
                        if any(ind in resp.lower() for ind in ['answer:', 'therefore', 'thus', 'hence']):
                            bonus += 0.1
                        if any(ind in resp.lower() for ind in ['=', '+', '-', '*', '/']):
                            bonus += 0.1
                        enhanced_advantages.append(base + bonus)
                    weights = softmax([adv/TEMPERATURE for adv in enhanced_advantages])
                    selected_index_in_valid = np.random.choice(len(weights), p=weights)
                    return valid_indices[selected_index_in_valid]

                # Select from the largest cluster
                cluster_sizes = [len(c) for c in clusters]
                largest_idx = int(np.argmax(cluster_sizes))
                selected_cluster = clusters[largest_idx]  # indices in valid_responses

                # Enhanced advantages in selected cluster
                enhanced_advantages = []
                for ddi in selected_cluster:
                    resp = valid_responses[ddi]
                    base_adv = valid_advantages[ddi]
                    bonus = 0
                    if len(resp.strip()) > 50:
                        bonus += 0.2
                    if any(ind in resp.lower() for ind in ['answer:', 'therefore', 'thus', 'hence']):
                        bonus += 0.1
                    if any(ind in resp.lower() for ind in ['=', '+', '-', '*', '/']):
                        bonus += 0.1
                    enhanced_advantages.append(base_adv + bonus)

                weights = softmax([adv/TEMPERATURE for adv in enhanced_advantages])
                selected_index_in_cluster = np.random.choice(len(weights), p=weights)
                selected_index_in_valid = selected_cluster[selected_index_in_cluster]
                selected_index_final = valid_indices[selected_index_in_valid]

                print(f'Selected from largest cluster (size: {len(selected_cluster)}) with enhanced advantage')
                return selected_index_final

            except Exception as e:
                print(f'Cannot select response based on cluster: {e}, using advantage-based selection with semantic bonus')
                enhanced_advantages = []
                for resp, base in zip(responses, advantages):
                    bonus = 0
                    if resp.strip() and len(resp.strip()) > 50:
                        bonus += 0.2
                    if any(ind in resp.lower() for ind in ['answer:', 'therefore', 'thus', 'hence']):
                        bonus += 0.1
                    if any(ind in resp.lower() for ind in ['=', '+', '-', '*', '/']):
                        bonus += 0.1
                    enhanced_advantages.append(base + bonus)
                weights = softmax([adv/TEMPERATURE for adv in enhanced_advantages])
                return np.random.choice(len(weights), p=weights)

        else:
            raise ValueError(f"Unknown strategy: {self.args.strategy}")

    def process_example(self, example, system_prompt):
        """
        Process a single example through the phi-decoding pipeline
        Args:
            example: Input example containing question and other fields
            system_prompt: System prompt for the model
        Returns:
            Dictionary containing results and statistics
        """
        # Initialize tracking variables
        token_stats = {"input": 0, "output": 0}
        rollout_stats = {"total": 0, "saved": 0}

        # Initialize trajectory pools
        traj_pool = [[] for _ in range(self.args.num_foresight)]
        step_pool = [[] for _ in range(self.args.num_foresight)]
        prob_pool = [[] for _ in range(self.args.num_foresight + 1)]
        adv_pool = [[] for _ in range(self.args.num_foresight + 1)]

        # Initialize beam states
        previous_steps = ["The reasoning steps are:\n\n" for _ in range(
            self.args.step_beam_size)]
        previous_values = [0.0 for _ in range(self.args.step_beam_size)]

        # Initialize trajectory information

        traj_info = {
            'question_idx': example.get('id', 0),
            'passage': example['passage'],
            'question': example['question'],
            'ground_truth': example.get('answer'),
            'foresight_part': [],  # Will be filled during each step
            'final_part': {},      # Will be filled during final generation
            'config': {            # Add configuration information
                'num_rollout': self.args.num_rollout,
                'num_foresight': self.args.num_foresight,
                'step_beam_size': self.args.step_beam_size,
                'strategy': self.args.strategy,
                'width_pruning_strategy': self.args.width_pruning_strategy,
                'depth_pruning_strategy': self.args.depth_pruning_strategy,
                'threshold': self.args.threshold,
                'sigma_rate': self.args.sigma_rate,
                'cluster_num': self.args.cluster_num
            }
        }

        # Multi-step reasoning
        for step in range(self.args.num_foresight):
            step_results = self._process_step(
                example,
                system_prompt,
                previous_steps,
                previous_values,
                token_stats,
                rollout_stats,
                traj_info  # Pass trajectory information
            )

            # Check early stopping condition
            if self._should_stop_early(step_results, step):
                break

            # Update state for next step
            previous_steps = step_results["next_steps"]
            previous_values = step_results["next_values"]

            # Record step results
            traj_pool[step] = step_results["trajectories"]
            step_pool[step] = step_results["steps"]
            prob_pool[step] = step_results["logprobs"]
            adv_pool[step] = step_results["advantages"]

        # Generate final response
        final_result = self._generate_final_response(
            example,
            system_prompt,
            previous_steps,
            previous_values,
            token_stats,
            rollout_stats,
            traj_info  # Pass trajectory information
        )

        # Record token statistics
        traj_info['token_num'] = token_stats["input"] + token_stats["output"]

        return {
            # "response_in_the_final_generation": final_result["response_in_the_final_generation"],
            "response": final_result["response"],
            "token_stats": token_stats,
            "rollout_stats": rollout_stats,
            "trajectories": {
                "steps": step_pool,
                "probs": prob_pool,
                "advantages": adv_pool,
                "final": final_result["trajectories"]
            },
            "traj_info": traj_info  # Add trajectory information to return result
        }

    def _process_step(self, example, system_prompt, previous_steps, previous_values, token_stats, rollout_stats, traj_info):
        """Process a single reasoning step"""
        stop_foresight = False
        # first stage: generate incomplete responses
        all_inputs = []
        for beam_idx in range(self.args.step_beam_size):
            chat = self._prepare_chat_template_for_first_stage(
                example, system_prompt)
            if self.args.model_id == "mistral":
                chat[1]['content'] = system_prompt + "\n" + chat[1]['content']
                chat = chat[1:]
            inputs = self.tokenizer.apply_chat_template(
                chat,
                tokenize=False
            ).rstrip(self.tokenizer.eos_token).rstrip()
            inputs = inputs + previous_steps[beam_idx]
            token_stats["input"] += len(self.tokenizer(inputs)["input_ids"])
            all_inputs.append(inputs)

        sampling_params = SamplingParams(
            max_tokens=1024, n=self.args.num_rollout, logprobs=0, temperature=0.4, stop=["\n", "<end_of_reasoning>"])

        outputs = self.model.generate(
            all_inputs,
            sampling_params
        )

        rollout_stats["total"] += self.args.num_rollout * \
            self.args.step_beam_size

        # collect the results of the first stage
        all_responses_first_stage = []
        all_logprobs_first_stage = []
        all_advantages_first_stage = []
        all_token_nums_first_stage = []

        for beam_idx, beam_outputs in enumerate(outputs):
            for output in beam_outputs.outputs:
                response = output.text.strip()
                logprob = output.cumulative_logprob / \
                    (len(output.token_ids) + 1e-8)
                advantage = logprob - previous_values[beam_idx]

                all_responses_first_stage.append(response)
                all_logprobs_first_stage.append(logprob)
                all_advantages_first_stage.append(advantage)
                all_token_nums_first_stage.append(len(output.token_ids))
                token_stats["output"] += len(output.token_ids)

        # Debug: Print all first stage responses
        print(f"\n=== First Stage Responses (Total: {len(all_responses_first_stage)}) ===")
        for i, response in enumerate(all_responses_first_stage):
            print(f"Sample {i}: {response}")
        print("=" * 50)

        # prune the responses based on width pruning_strategy
        if self.args.width_pruning_strategy != "none" and self.args.width_pruning_strategy != "":
            keep_foresight_list = []
            if self.args.width_pruning_strategy == "low_sigma":
                # calculate the mean and standard deviation of logprobs
                mean = np.mean(all_logprobs_first_stage)
                std = np.std(all_logprobs_first_stage)

                # keep the samples with logprob higher than mean - sigma_rate * std
                for idx, logp in enumerate(all_logprobs_first_stage):
                    if logp > mean - self.args.sigma_rate * std:
                        keep_foresight_list.append(idx)
                
                # Preserve high semantic quality responses even if logprob is low
                semantic_quality_indices = []
                for idx, response in enumerate(all_responses_first_stage):
                    if idx not in keep_foresight_list:  # Only check responses not already kept
                        semantic_score = 0
                        if len(response.strip()) > 50:  # Length bonus
                            semantic_score += 0.2
                        if any(ind in response.lower() for ind in ['answer:', 'therefore', 'thus', 'hence']):  # Reasoning indicators
                            semantic_score += 0.1
                        if any(ind in response.lower() for ind in ['=', '+', '-', '*', '/']):  # Mathematical operations
                            semantic_score += 0.1
                        if semantic_score >= 0.3:  # High semantic quality threshold
                            semantic_quality_indices.append(idx)
                
                # Add high semantic quality responses to keep list
                keep_foresight_list.extend(semantic_quality_indices)
                keep_foresight_list = list(set(keep_foresight_list))  # Remove duplicates

            # if the number of kept samples is less than step_beam_size, then supplement
            if len(keep_foresight_list) < self.args.step_beam_size:
                weights = softmax(
                    [logp/TEMPERATURE for logp in all_logprobs_first_stage])
                num_to_add = self.args.step_beam_size - \
                    len(keep_foresight_list)
                available_indices = [i for i in range(
                    len(all_logprobs_first_stage)) if i not in keep_foresight_list]
                if available_indices:
                    available_weights = [weights[i] for i in available_indices]
                    available_weights = [w/sum(available_weights)
                                         for w in available_weights]
                    additional_indices = np.random.choice(
                        available_indices,
                        # size=min(num_to_add, len(available_indices)),
                        size=num_to_add,
                        p=available_weights,
                        replace=False
                    ).tolist()
                    keep_foresight_list.extend(additional_indices)

            keep_foresight_list.sort()

            # update the statistics
            rollout_stats["saved"] += (self.args.step_beam_size *
                                       self.args.num_rollout - len(keep_foresight_list))

            # only keep the selected samples
            filtered_responses = [all_responses_first_stage[i]
                                  for i in keep_foresight_list]
            filtered_logprobs = [all_logprobs_first_stage[i]
                                 for i in keep_foresight_list]
            filtered_advantages = [all_advantages_first_stage[i]
                                   for i in keep_foresight_list]
            filtered_beam_indices = [
                i // self.args.num_rollout for i in keep_foresight_list]

            all_responses = filtered_responses

            # Debug: Print filtered responses after width pruning
            print(f"\n=== Filtered Responses After Width Pruning (Total: {len(filtered_responses)}) ===")
            for i, response in enumerate(filtered_responses):
                print(f"Filtered {i}: {response}")
            print("=" * 50)

        # second stage: complete the responses
        completion_inputs = []
        for idx in range(len(keep_foresight_list)):
            response = all_responses[idx]
            # if response.strip() != '':
            chat = self._prepare_chat_template(example, system_prompt)
            beam_idx = keep_foresight_list[idx] // self.args.num_rollout
            chat[-1]["content"] = previous_steps[beam_idx] + response

            inputs = self.tokenizer.apply_chat_template(
                chat,
                tokenize=False
            ).rstrip(self.tokenizer.eos_token).rstrip()

            completion_inputs.append(inputs)
            token_stats["input"] += len(self.tokenizer(inputs)
                                        ["input_ids"])

        # generate the completed responses
        sampling_params = SamplingParams(
            max_tokens=1024,
            n=1,
            logprobs=0,
            stop=["<end_of_reasoning>"]
        )

        completion_outputs = self.model.generate(
            completion_inputs, sampling_params)
        rollout_stats["total"] += len(completion_inputs)

        # collect the results of the second stage
        completed_responses = []
        completed_logprobs = []
        completed_advantages = []

        for idx, outputs in enumerate(completion_outputs):
            output = outputs.outputs[0]
            response = output.text.strip()
            logprob = output.cumulative_logprob / \
                (len(output.token_ids) + 1e-8)
            beam_idx = keep_foresight_list[idx] // self.args.num_rollout
            advantage = logprob - previous_values[beam_idx]

            completed_responses.append(response)
            completed_logprobs.append(logprob)
            completed_advantages.append(advantage)
            token_stats["output"] += len(output.token_ids)

        # Debug: Print completed responses after second stage
        print(f"\n=== Completed Responses After Second Stage (Total: {len(completed_responses)}) ===")
        for i, response in enumerate(completed_responses):
            print(f"Completed {i}: {response}")
        print("=" * 50)

        # third stage: cluster and select the completed responses
        try:
            # Use improved clustering with robustness checks
            clusters, cluster_info = self.cluster_responses(completed_responses, completed_advantages)
            
            if clusters is None:
                print(f'Clustering failed in step: {cluster_info["reason"]}, using advantage-based selection')
                weights = softmax([adv/TEMPERATURE for adv in completed_advantages])
                selected = np.random.choice(
                    len(weights),
                    size=self.args.step_beam_size,
                    p=weights,
                    replace=False
                ).tolist()
                stop_foresight = False
            else:
                # build the cluster list
                cluster_list = [[] for _ in range(self.args.cluster_num)]
                for idx, cluster in enumerate(clusters):
                    cluster_list[idx] = cluster
                cluster_list = [sorted(cluster) for cluster in cluster_list]

                # calculate the cluster weights and advantage weights
                cluster_len_ratio = [len(cluster)/len(completed_responses)
                                     for cluster in cluster_list]
                per_sample_cluster_len_ratio = []
                for i in range(len(completed_responses)):
                    for cluster_idx, cluster in enumerate(cluster_list):
                        if i in cluster:
                            per_sample_cluster_len_ratio.append(cluster_len_ratio[cluster_idx])
                            break
                
                cluster_weights = softmax(per_sample_cluster_len_ratio)
                adv_weights = softmax([adv/TEMPERATURE for adv in completed_advantages])

                # combine the weights
                weights = [(cluster_weights[ii] + adv_weights[ii]) /
                           2 for ii in range(len(completed_responses))]

                # select the samples
                selected = np.random.choice(
                    len(weights),
                    size=self.args.step_beam_size,
                    p=weights,
                    replace=False
                ).tolist()

                # Check if largest cluster ratio exceeds threshold
                largest_cluster_size = max(len(cluster) for cluster in cluster_list)
                largest_ratio = largest_cluster_size / len(completed_responses)

                if largest_ratio >= self.args.threshold:
                    stop_foresight = True

            # Record information after generating first stage responses
            step_info = {
                'first_stage': {
                    'responses': all_responses_first_stage,
                    'logprobs': all_logprobs_first_stage,
                    'advantages': all_advantages_first_stage,
                    'token_nums': all_token_nums_first_stage
                }
            }

            # Record information after width pruning
            if self.args.width_pruning_strategy != "none" and self.args.width_pruning_strategy != "":
                step_info['width_pruning'] = {
                    'keep_indices': keep_foresight_list,
                    'filtered_responses': filtered_responses,
                    'filtered_logprobs': filtered_logprobs,
                    'filtered_advantages': filtered_advantages,
                    'filtered_beam_indices': filtered_beam_indices
                }

            # Record information after second stage completion
            step_info['second_stage'] = {
                'responses': completed_responses,
                'logprobs': completed_logprobs,
                'advantages': completed_advantages
            }

            # Record information after clustering and selection
            if clusters is not None:
                step_info['clustering'] = {
                    'cluster_sizes': [len(cluster) for cluster in cluster_list],
                    'cluster_weights': cluster_weights.tolist(),
                    'adv_weights': adv_weights.tolist(),
                    'combined_weights': weights,
                    'selected_indices': selected
                }
            else:
                step_info['clustering'] = {
                    'state': 'failed',
                    'reason': cluster_info.get('reason', 'unknown')
                }

            # Record final selection results
            step_info['final'] = {
                'selected_steps': [previous_steps[keep_foresight_list[idx]//self.args.num_rollout] + all_responses_first_stage[keep_foresight_list[idx]] + "\n" for idx in selected],
                'selected_values': [completed_logprobs[idx] for idx in selected],
                'selected_indices': selected
            }

            # Debug: Print final selected responses
            print(f"\n=== Final Selected Responses (Total: {len(selected)}) ===")
            for i, idx in enumerate(selected):
                print(f"Selected {i} (idx {idx}): {completed_responses[idx]}")
            print("=" * 50)

            # Add current step information to trajectory information
            traj_info['foresight_part'].append(step_info)

            return {
                "next_steps": [previous_steps[keep_foresight_list[idx]//self.args.num_rollout] + all_responses_first_stage[keep_foresight_list[idx]] + "\n" for idx in selected],
                "next_values": [completed_logprobs[idx] for idx in selected],
                "trajectories": completed_responses,
                "steps": [keep_foresight_list[idx] for idx in selected],
                "logprobs": completed_logprobs,
                "advantages": completed_advantages,
                "stop_foresight": stop_foresight
            }

        except Exception as e:
            print(
                'when cluster during intermediate steps, error occurs, use adv no replace')
            weights = softmax(
                [adv/TEMPERATURE for adv in completed_advantages])
            selected = np.random.choice(
                len(weights),
                size=self.args.step_beam_size,
                p=weights,
                replace=False
            ).tolist()

            return {
                "next_steps": [previous_steps[keep_foresight_list[idx]//self.args.num_rollout] + all_responses_first_stage[keep_foresight_list[idx]] + "\n" for idx in selected],
                "next_values": [all_logprobs_first_stage[idx] for idx in selected],
                "trajectories": all_responses_first_stage,
                "steps": [keep_foresight_list[idx] for idx in selected],
                "logprobs": all_logprobs_first_stage,
                "advantages": all_advantages_first_stage,
                "stop_foresight": stop_foresight
            }

    def _should_stop_early(self, step_results, current_step):
        """Check if early stopping conditions are met with enhanced quality assessment"""
        if current_step < self.args.least_foresight_num:
            return False

        # Check if all responses are identical (original condition)
        just_stop = True
        first_response = step_results["trajectories"][0]
        for response in step_results["trajectories"][1:]:
            if response != first_response:
                just_stop = False
                break

        if just_stop:
            print(
                f'Early stopping at depth {current_step} (all responses are the same)')
            return True

        # Enhanced quality assessment: check if average advantage is too low
        if hasattr(step_results, 'advantages') and step_results['advantages']:
            avg_advantage = np.mean(step_results['advantages'])
            if avg_advantage < -2.0:  # Threshold for poor reasoning quality
                print(f'Early stopping at depth {current_step} (poor reasoning quality, avg advantage: {avg_advantage:.3f})')
                return True

        # Check for repetitive patterns (high similarity between responses)
        if len(step_results["trajectories"]) > 1:
            from difflib import SequenceMatcher
            similarity_scores = []
            for i in range(len(step_results["trajectories"])):
                for j in range(i+1, len(step_results["trajectories"])):
                    similarity = SequenceMatcher(None, 
                                               step_results["trajectories"][i], 
                                               step_results["trajectories"][j]).ratio()
                    similarity_scores.append(similarity)
            
            if similarity_scores and np.mean(similarity_scores) > 0.8:  # High similarity threshold
                print(f'Early stopping at depth {current_step} (high response similarity: {np.mean(similarity_scores):.3f})')
                return True

        if self.args.depth_pruning_strategy == "cluster":
            # Check if responses are becoming similar
            if step_results["stop_foresight"]:
                print(
                    f'Early stopping at depth {current_step} (max cluster ratio >= args.threshold)')
                return True

        return False

    def _generate_final_response(self, example, system_prompt, previous_steps, previous_values, token_stats, rollout_stats, traj_info):
        """Generate final response after multi-step reasoning"""
        # Prepare input for each beam
        all_inputs = []
        for beam_idx in range(self.args.step_beam_size):
            chat = self._prepare_chat_template(example, system_prompt)
            chat[-1]["content"] = previous_steps[beam_idx]

            inputs = self.tokenizer.apply_chat_template(
                chat,
                tokenize=False
            ).rstrip(self.tokenizer.eos_token).rstrip()

            token_stats["input"] += len(self.tokenizer(inputs)["input_ids"])
            all_inputs.append(inputs)

        # parallel generate all beam responses
        sampling_params = SamplingParams(
            max_tokens=3000,
            n=1,
            logprobs=0,
            stop=["<end_of_reasoning>"]
        )
        outputs = self.model.generate(all_inputs, sampling_params)

        rollout_stats["total"] += self.args.step_beam_size

        # Collect all response results
        all_responses = []
        all_logprobs = []
        all_advantages = []
        all_combined_responses = []

        for beam_idx, beam_outputs in enumerate(outputs):
            output = beam_outputs.outputs[0]
            response = output.text.strip()
            logprob = output.cumulative_logprob / \
                (len(output.token_ids) + 1e-8)
            advantage = logprob - previous_values[beam_idx]

            # Combine previous_steps and new response
            combined_response = previous_steps[beam_idx] + response
            all_combined_responses.append(combined_response)
            all_responses.append(response)
            all_logprobs.append(logprob)
            all_advantages.append(advantage)
            token_stats["output"] += len(output.token_ids)

        # Debug: Print final stage responses
        print(f"\n=== Final Stage Responses (Total: {len(all_responses)}) ===")
        for i, response in enumerate(all_responses):
            print(f"Final {i}: {response}")
        print("=" * 50)

        # Select final response
        selected_idx = self.select_response(
            all_responses,
            all_logprobs,
            all_advantages
        )

        # Debug: Print final selected response
        print(f"\n=== Final Selected Response ===")
        print(f"Selected index: {selected_idx}")
        print(f"Selected response: {all_responses[selected_idx]}")
        print(f"Combined response: {all_combined_responses[selected_idx]}")
        print("=" * 50)

        # Record final results
        traj_info['final_part']['responses'] = all_combined_responses
        traj_info['final_part']['responses_in_the_final_generation'] = all_responses
        traj_info['final_part']['logprobs'] = all_logprobs
        traj_info['final_part']['advantages'] = all_advantages
        traj_info['final_part']['selected_idx'] = selected_idx

        return {
            "response": previous_steps[selected_idx] + all_responses[selected_idx],
            # "response_in_the_final_generation": all_responses[selected_idx],
            "trajectories": {
                "responses": all_responses,
                "logprobs": all_logprobs,
                "advantages": all_advantages,
                "selected_idx": selected_idx
            }
        }

    def _prepare_chat_template(self, example, system_prompt):
        """
        Prepare chat template based on dataset type
        Args:
            example: Input example
            system_prompt: System prompt
        Returns:
            List of chat messages
        """
        passage = example['passage']
        question = example['question']
        chat = [
            {'role': 'system', 'content': system_prompt},
            {'role': 'user', 'content': 'Passage: ' + passage + '\nQuestion: ' + question + '\nPlease directly follow the previous reasoning steps (if provided) and generate the remaining ones.\n'},
            {'role': 'assistant', 'content': ''}
        ]
        return chat

    def _prepare_chat_template_for_first_stage(self, example, system_prompt):
        """
        Prepare chat template based on dataset type
        Args:
            example: Input example
            system_prompt: System prompt
        Returns:
            List of chat messages
        """
        passage = example['passage']
        question = example['question']
        chat = [
            {'role': 'system', 'content': system_prompt},
            {'role': 'user', 'content': 'Passage: ' + passage + '\nQuestion: ' + question + '\nPlease directly follow the previous reasoning steps (if provided) and generate the remaining ones.\n'},
            {'role': 'assistant', 'content': ''}
        ]
        return chat


def main():
    """Main execution function"""
    args = parse_arguments()
    decoder = PhiDecoder(args)

    with open(args.data_path) as f:
        test_data = json.load(f)
    max_num = len(test_data) if args.max_examples == -1 else min(len(test_data), args.max_examples)
    os.makedirs(args.output_dir, exist_ok=True)
    os.makedirs(args.time_path, exist_ok=True)
    output_path = os.path.join(args.output_dir, f"{args.file_name}.json")

    # Record start time
    start_time = time.time()

    # Statistics
    total_stats = {
        "total_rollouts": 0,
        "saved_rollouts": 0,
        "input_tokens": 0,
        "output_tokens": 0
    }

    # Used to store all trajectory information
    all_traj_info = []

    # Process each test example
    for i, example in enumerate(test_data[:max_num]):
        print(f"正在处理第{i+1}/{max_num}个样本，问题：{example.get('question', '')[:50]}")
        # try:
        # Generate system prompt
        system_prompt = decoder.get_system_prompt()

        # Process example
        result = decoder.process_example(example, system_prompt)

        # Update statistics
        total_stats["total_rollouts"] += result["rollout_stats"]["total"]
        total_stats["saved_rollouts"] += result["rollout_stats"]["saved"]
        total_stats["input_tokens"] += result["token_stats"]["input"]
        total_stats["output_tokens"] += result["token_stats"]["output"]

        # Add trajectory information
        result["traj_info"]["question_idx"] = i
        all_traj_info.append(result["traj_info"])

        # Prepare output result
        output_result = {
            "id": i,
            "question": example["question"],
            "passage": example["passage"],
            "ground_truth": example.get("answer"),
            "response": result["response"],
            "response_all_beams": result["trajectories"]["final"]["responses"] if "final" in result["trajectories"] else []
        }

        # Write result to main output file
        with open(output_path, "a") as f:
            f.write(json.dumps(output_result) + "\n")

        print(
            f'output_token_num_for_question{i}: {result["token_stats"]["output"]}')
        print(
            f'input_token_num_for_question{i}: {result["token_stats"]["input"]}')
        print(f'all_output_token_num: {total_stats["output_tokens"]}')
        print(f'all_input_token_num: {total_stats["input_tokens"]}')

        # Save trajectory information
        if args.record_process:
            traj_path = os.path.join(
                args.time_path, f"TRAJ_INFO-{args.file_name}.json")
            with open(traj_path, "w") as f:
                json.dump(all_traj_info, f, indent=2)

    # Calculate total time
    end_time = time.time()
    time_span = end_time - start_time

    # Save time information to separate file
    time_info_path = os.path.join(args.time_path, f"{args.file_name}.txt")
    with open(time_info_path, "w") as f:
        f.write(f'time:  {time_span}\n')
        f.write(f'total:  {total_stats["total_rollouts"]}\n')
        f.write(f'save:  {total_stats["saved_rollouts"]}\n')
        f.write(f'num_rollout:  {args.num_rollout}\n')
        f.write(f'num_foresight:  {args.num_foresight}\n')
        f.write(f'step_beam_size:  {args.step_beam_size}\n')
        f.write(f'strategy:  {args.strategy}\n')
        f.write(f'width_pruning_strategy:  {args.width_pruning_strategy}\n')
        f.write(f'depth_pruning_strategy:  {args.depth_pruning_strategy}\n')
        f.write(f'threshold:  {args.threshold}\n')
        f.write(f'sigma_rate:  {args.sigma_rate}\n')
        f.write(f'cluster_num:  {args.cluster_num}\n')
        f.write(f'all_input_token_num:  {total_stats["input_tokens"]}\n')
        f.write(f'all_output_token_num:  {total_stats["output_tokens"]}\n')

    print('total rollouts: ', total_stats["total_rollouts"])
    print('saved rollouts: ', total_stats["saved_rollouts"])
    print('all_output_token_num: ', total_stats["output_tokens"])
    print('all_input_token_num: ', total_stats["input_tokens"])


if __name__ == "__main__":
    main()
