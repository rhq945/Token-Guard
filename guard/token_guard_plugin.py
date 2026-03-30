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
# --- Transformers-only backend ---
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    StoppingCriteriaList,
    StoppingCriteria
)

# --- [2] Token Guard Imports ---
from token_guard_plugin import LatentEnvironment, TokenGuardConfig, CandidateSegment

# --- [3] Data Imports (假设 logic_example.py 在同一目录下) ---
from logic_example import (
    HISTORY_8_FEW_SHOT,
    NFL_8_FEW_SHOT,
    halueval_6_FEW_SHOT,
    covidQA_4_FEW_SHOT,
    financebench_5_FEW_SHOT,
    pubmedQA_4_FEW_SHOT,
    RAGTruth_5_FEW_SHOT
)

# Constants for algorithm configuration
INF = 10  # Used for initialization of min/max values
TEMPERATURE = 0.3  # Temperature parameter for softmax sampling (restored to baseline optimal)
# GEN_TEMPERATURE = 0.7    # 生成温度：调高！让模型敢于探索不同的措辞和思路
# SELECT_TEMPERATURE = 1.0 # 选择温度：调高！防止 Softmax 过于激进，保留更多候选的可能性


def parse_arguments():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description="TokenGuard Decoding Algorithm")

    # Model configuration
    parser.add_argument('--model_id', type=str, default='llama3.1',
                        help='Model identifier')
    parser.add_argument('--model_path', type=str, default='/data/rhq/TOKEN-GUARD/halu/models/Meta-Llama-3.1-8B-Instruct',
                        help='Model path')
    parser.add_argument('--gpus', type=int, default=1,
                        help='Number of GPUs to use')

    # Data configuration
    parser.add_argument('--datasets', type=str, default='gsm',
                        help='Dataset type')  # gsm, math, reclor, logiqa, gpqa, arc
    parser.add_argument('--data_path', type=str,
                        default='/data/rhq/TOKEN-GUARD/halu/data2/halueval.json',
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
    x = np.array(x)
    # [修复] 处理 NaN 和 Inf
    x = np.nan_to_num(x, nan=-1e9, posinf=1e9, neginf=-1e9)
    # 稳定性技巧：平移数值
    x_max = np.max(x)
    if x_max == -1e9: # 如果全是 NaN
        return np.ones(len(x)) / len(x) # 返回均匀分布
        
    e_x = np.exp(x - x_max)
    return e_x / e_x.sum(axis=0)


class TokenGuardDecoder:
    def __init__(self, args):
        """
        Initialize the decoder
        Args:
            args: Command line arguments containing configuration
        """
        self.args = args
        self.model = None
        self.tokenizer = None
        self.tg_scorer = None  # TokenGuard Scorer
        self.initialize_model()

    def initialize_model(self):
        """
        Initialize the model using LatentEnvironment to allow SHARED MEMORY.
        We do NOT use VLLM here to avoid double loading.
        """
        print(f"Loading Shared Model via TokenGuard from {self.args.model_path}...")
        
        # 1. Initialize Token-Guard (This loads the HF Model onto GPU)
        # We assume the user has enough VRAM for one instance of the model.
        # tg_config = TokenGuardConfig(device="cpu") 
        tg_config = TokenGuardConfig(device="cuda") 
        self.tg_scorer = LatentEnvironment(
            model_path=self.args.model_path, 
            config=tg_config
        )
        
        # 2. Reuse the model and tokenizer from TokenGuard for generation
        # This achieves the "Single Model" requirement.
        self.model = self.tg_scorer.model
        self.tokenizer = self.tg_scorer.tokenizer
        
        # 3. Ensure pad_token is set (Critical for HF generation)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
            self.model.config.pad_token_id = self.tokenizer.eos_token_id

        print(f"[TokenGuard] Generator mode: Local HuggingFace")
        self.model = self.tg_scorer.model
        self.model.config.pad_token_id = self.tokenizer.eos_token_id

        self.tg_lambda = 2.0 
        np.random.seed(self.args.seed)
        torch.manual_seed(self.args.seed)

    def _get_model_path(self):
        """Get the appropriate model path"""
        return self.args.model_path

    def _generate(self, prompts: List[str], n_return: int, max_new_tokens: int = 1024, stop_strs: List[str] = None) -> Tuple[List[str], List[float]]:
        """
        Generates text using the shared HF model with robust error handling.
        Fix: Corrected padding logic to prevent index overflow.
        """
        responses = []
        cumulative_logprobs = []
        
        # Ensure pad token is set
        if self.tokenizer.pad_token_id is None:
            self.tokenizer.pad_token_id = self.tokenizer.eos_token_id
            self.model.config.pad_token_id = self.tokenizer.eos_token_id

        for prompt_idx, prompt in enumerate(prompts):
            # 记录当前 prompt 开始前的结果数量
            start_count = len(responses)
            
            # Truncate input
            inputs = self.tokenizer(
                prompt, 
                return_tensors="pt", 
                truncation=True, 
                max_length=2048 
            ).to(self.model.device)
            
            input_len = inputs.input_ids.shape[1]
            outputs = None 

            with torch.no_grad():
                try:
                    # Try Sampling
                    outputs = self.model.generate(
                        **inputs,
                        max_new_tokens=max_new_tokens,
                        do_sample=True,
                        temperature=0.7, # [关键修改] 调高温度！不要用 0.3，改成 0.7 或 0.9
                        top_p=0.95, 
                        top_k=50, 
                        repetition_penalty=1.0, 
                        num_return_sequences=n_return,
                        return_dict_in_generate=True, 
                        output_scores=True,           
                        pad_token_id=self.tokenizer.pad_token_id,
                        eos_token_id=self.tokenizer.eos_token_id
                    )
                except RuntimeError as e:
                    print(f"⚠️ Generation Error: {e}. Falling back to greedy decoding.")
                    # Fallback to Greedy
                    outputs = self.model.generate(
                        **inputs,
                        max_new_tokens=max_new_tokens,
                        do_sample=False,       
                        num_return_sequences=1, 
                        return_dict_in_generate=True, 
                        output_scores=True,           
                        pad_token_id=self.tokenizer.pad_token_id,
                        eos_token_id=self.tokenizer.eos_token_id
                    )
                    
                    # Manually duplicate for greedy fallback
                    if n_return > 1:
                        outputs.sequences = outputs.sequences.repeat(n_return, 1)
                        if outputs.scores:
                            new_scores = []
                            for step_score in outputs.scores:
                                new_scores.append(step_score.repeat(n_return, 1))
                            outputs.scores = tuple(new_scores)

            # Calculate Logprobs
            if outputs is not None and outputs.scores:
                transition_scores = self.model.compute_transition_scores(
                    outputs.sequences, outputs.scores, normalize_logits=True
                )
            else:
                transition_scores = torch.zeros((n_return, 1)).to(self.model.device)

            actual_return_num = outputs.sequences.shape[0]
            
            for i in range(actual_return_num):
                generated_ids = outputs.sequences[i][input_len:]
                text = self.tokenizer.decode(generated_ids, skip_special_tokens=True)
                
                if stop_strs:
                    for s in stop_strs:
                        if s in text:
                            text = text.split(s)[0]
                            break
                
                responses.append(text)
                
                if transition_scores.numel() > 0 and i < transition_scores.shape[0]:
                    valid_len = min(len(generated_ids), transition_scores.shape[1])
                    seq_logprob = transition_scores[i, :valid_len].sum().item()
                    avg_logprob = seq_logprob / (valid_len + 1e-8)
                else:
                    avg_logprob = -1.0
                cumulative_logprobs.append(avg_logprob)
                
            # [关键修复] 这里的补齐逻辑只针对当前 Prompt
            # 目标是当前 prompt 必须贡献 n_return 个结果
            expected_total = start_count + n_return
            while len(responses) < expected_total:
                 responses.append(responses[-1] if len(responses) > start_count else "")
                 cumulative_logprobs.append(-1.0)

        return responses, cumulative_logprobs
        
    def get_system_prompt(self, dataset_type=None):
        """
        Get the appropriate system prompt based on dataset type
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
                "Answer the user's question strictly based on the given information.\n"
                "Do not make up information.\n"
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

    def cluster_and_select_chains(self, responses, advantages, data_size: int = None):
        """
        Cluster reasoning chains using TF-IDF and KMeans.
        K=5 for large datasets (>100 samples), K=3 for small datasets (<=100).
        Falls back to self.args.cluster_num when data_size is not provided.
        (PAPER_STORY.md Stage 3: cluster_and_select_chains)
        """
        # Dynamic K selection per paper spec (K=5 large, K=3 small)
        if data_size is not None:
            k = 5 if data_size > 100 else 3
        else:
            k = self.args.cluster_num   # CLI fallback

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
                max_features=500,
                min_df=1,
                max_df=0.9,
                ngram_range=(1, 1)
            )
            X = vectorizer.fit_transform(valid_responses)

            # Check if we have enough features for clustering
            if X.shape[1] < 2:
                return None, {"state": "cannot cluster", "reason": "insufficient features"}

            # Cap k to the number of distinct valid responses
            k = min(k, len(set(valid_responses)))

            # Perform clustering with silhouette analysis
            kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
            kmeans.fit(X)

            # Calculate silhouette score to assess clustering quality
            if len(valid_responses) > 2 and len(set(kmeans.labels_)) > 1:
                silhouette_avg = silhouette_score(X, kmeans.labels_)
                if silhouette_avg < 0.05:
                    return None, {"state": "cannot cluster", "reason": f"poor clustering quality (silhouette: {silhouette_avg:.3f})"}
            else:
                silhouette_avg = 0.0

            # Group responses by cluster
            clusters = [[] for _ in range(k)]
            for idx, label in enumerate(kmeans.labels_):
                clusters[label].append(valid_indices[idx])

            return clusters, {
                "state": "success",
                "k": k,
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
                # Use higher temperature (1.0) for robustness
                weights = softmax([-adv/TEMPERATURE for adv in advantages])
                return np.random.choice(len(advantages), p=weights)

            if len(valid_indices) < self.args.step_beam_size:
                print('valid responses are less than step_beam_size, use adv no replace')
                # Use higher temperature (1.0) for robustness
                weights = softmax([adv/TEMPERATURE for adv in advantages])
                return np.random.choice(len(advantages), p=weights)

            try:
                # prepare cluster data (compress to valid items)
                valid_responses = [responses[i] for i in valid_indices]
                valid_advantages = [advantages[i] for i in valid_indices]

                # Use improved clustering with robustness checks
                clusters, cluster_info = self.cluster_and_select_chains(valid_responses, valid_advantages)
                
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
                    
                    # [Fix] Use temperature 1.0 for selection to prevent numerical instability
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

                # [Fix] Use temperature 1.0 for selection to prevent numerical instability
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
                # [Fix] Use temperature 1.0 for fallback as well
                weights = softmax([adv/TEMPERATURE for adv in enhanced_advantages])
                return np.random.choice(len(weights), p=weights)

        else:
            raise ValueError(f"Unknown strategy: {self.args.strategy}")

    def process_example(self, example, system_prompt):
        """
        Process a single example through the phi-decoding pipeline with Token-Guard Integration
        """
        # Initialize tracking variables
        token_stats = {"input": 0, "output": 0}
        rollout_stats = {"total": 0, "saved": 0}

        # =================================================================
        # === [Token-Guard] Step 1: Environment & History Init (Eq. 1) ===
        # =================================================================
        # 1. Construct initial Context
        init_context = f"{system_prompt}\nPassage: {example['passage']}\nQuestion: {example['question']}\n"
        
        # 2. Calculate initial anchor
        current_h_x = self.tg_scorer.initialize_anchor(init_context)
        
        # 3. Initialize Semantic History (List[List[CandidateSegment]])
        beam_artifacts = [[] for _ in range(self.args.step_beam_size)]
        # =================================================================

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
            'foresight_part': [],  
            'final_part': {},     
            'config': {            
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
            # [Token-Guard] Pass h_x and artifacts to _process_step
            step_results = self._process_step(
                example,
                system_prompt,
                previous_steps,
                previous_values,
                token_stats,
                rollout_stats,
                traj_info,
                current_h_x,    # <--- New
                beam_artifacts  # <--- New
            )

            # Check early stopping condition
            if self._should_stop_early(step_results, step):
                break

            # Update state for next step
            previous_steps = step_results["next_steps"]
            previous_values = step_results["next_values"]

            # =================================================================
            # === [Token-Guard] History Update & Dynamic Thresholds (Eq. 15) ===
            # =================================================================
            if "next_artifacts" in step_results:
                beam_artifacts = step_results["next_artifacts"]

                # Dynamic Threshold Adjustment: Based on Global Score of best Beam
                if beam_artifacts and len(beam_artifacts) > 0 and beam_artifacts[0]:
                    _, f_fact, f_logic = self.tg_scorer.compute_chain_global_score(beam_artifacts[0])
                    self.tg_scorer.adjust_thresholds(f_fact, f_logic)
            # =================================================================

            # Record step results
            traj_pool[step] = step_results["trajectories"]
            step_pool[step] = step_results["steps"]
            prob_pool[step] = step_results["logprobs"]
            adv_pool[step] = step_results["advantages"]

        # Step A: 从所有 beam_artifacts 中找出全局分最高的链
        best_chain = []
        best_global_score = -1.0
        for chain in beam_artifacts:
            if chain:
                f_g, _, _ = self.tg_scorer.compute_chain_global_score(chain)
                if f_g > best_global_score:
                    best_global_score = f_g
                    best_chain = list(chain)  # 浅拷贝，防止后续 refine 污染原始列表

        # Step B: M_max=2 迭代循环，目标是让 best_global_score >= τ_global=0.7
        cannot_answer = False
        for global_iter in range(self.tg_scorer.config.m_max):
            if best_global_score >= self.tg_scorer.config.tau_global:
                print(f"[GlobalIter {global_iter}] Converged: "
                      f"{best_global_score:.3f} >= τ_global={self.tg_scorer.config.tau_global:.3f}")
                break

            print(f"[GlobalIter {global_iter}] Score {best_global_score:.3f} "
                  f"< τ_global={self.tg_scorer.config.tau_global:.3f}. Refining worst segment...")

            if not best_chain:
                cannot_answer = True
                break

            # Step C: 根据 Fact/Logic 分歧动态调整阈值（Eq. 15, Δτ=0.1）
            _, f_fact, f_logic = self.tg_scorer.compute_chain_global_score(best_chain)
            self.tg_scorer.adjust_thresholds(f_fact, f_logic)

            # Step D: 定位链中分数最低的片段并局部精化（1次轻量 refine，控制耗时）
            worst_seg_idx = min(range(len(best_chain)),
                                key=lambda k: best_chain[k].segment_score)
            init_context = (f"{system_prompt}\n"
                            f"Passage: {example['passage']}\n"
                            f"Question: {example['question']}\n")
            refined_seg = self.tg_scorer.refine_segment(
                context_text=init_context,
                artifact=best_chain[worst_seg_idx],
                h_x=current_h_x,
                max_retries=1   # 全局迭代内每次只精化 1 步
            )
            best_chain[worst_seg_idx] = refined_seg

            # Step E: 用精化后的链重新计算全局分
            best_global_score, _, _ = self.tg_scorer.compute_chain_global_score(best_chain)

            # Step F: 最后一次迭代仍未收敛 → 触发 fallback
            if global_iter == self.tg_scorer.config.m_max - 1:
                if best_global_score < self.tg_scorer.config.tau_global:
                    cannot_answer = True
                    print(f"[GlobalIter] Failed after {self.tg_scorer.config.m_max} "
                          f"iterations (final score={best_global_score:.3f}). "
                          f"Outputting 'cannot answer'.")

        # Step G: Fallback — 全局迭代失败，输出 "cannot answer"
        if cannot_answer:
            traj_info['token_num'] = token_stats["input"] + token_stats["output"]
            return {
                "response": "cannot answer",
                "token_stats": token_stats,
                "rollout_stats": rollout_stats,
                "trajectories": {
                    "steps": step_pool,
                    "probs": prob_pool,
                    "advantages": adv_pool,
                    "final": {}
                },
                "traj_info": traj_info
            }
        # =================================================================

        # Generate final response
        final_result = self._generate_final_response(
            example,
            system_prompt,
            previous_steps,
            previous_values,
            token_stats,
            rollout_stats,
            traj_info
        )

        # Record token statistics
        traj_info['token_num'] = token_stats["input"] + token_stats["output"]

        return {
            "response": final_result["response"],
            "token_stats": token_stats,
            "rollout_stats": rollout_stats,
            "trajectories": {
                "steps": step_pool,
                "probs": prob_pool,
                "advantages": adv_pool,
                "final": final_result["trajectories"]
            },
            "traj_info": traj_info
        }

    def _process_step(self, example, system_prompt, previous_steps, previous_values, 
                        token_stats, rollout_stats, traj_info, 
                        current_h_x, beam_artifacts): # [新增参数]
            """Process a single reasoning step with Token-Guard Integration and HF Generation"""
            stop_foresight = False
            
            # ---------------------------------------------------------------
            # 1. First Stage: Generate responses (Simplified to single stage via HF)
            # ---------------------------------------------------------------
            # In VLLM version, we did (incomplete) -> prune -> (complete).
            # In HF version, we generate the full step at once for efficiency.
            all_prompts = []
            for beam_idx in range(self.args.step_beam_size):
                chat = self._prepare_chat_template(
                    example, system_prompt)
                if self.args.model_id == "mistral":
                    chat[1]['content'] = system_prompt + "\n" + chat[1]['content']
                    chat = chat[1:]
                base_prompt = self.tokenizer.apply_chat_template(
                    chat,
                    tokenize=False
                ).rstrip(self.tokenizer.eos_token).rstrip()
                full_prompt = base_prompt + previous_steps[beam_idx]
                all_prompts.append(full_prompt)
                token_stats["input"] += len(self.tokenizer(full_prompt)["input_ids"])

            # Use our HF helper function to generate
            # stop_strs mimics the VLLM behavior of stopping at newline for reasoning steps
            raw_responses, raw_logprobs = self._generate(
                all_prompts,
                n_return=self.args.num_rollout,
                max_new_tokens=1024,
                stop_strs=["\n", "<end_of_reasoning>"]
            )
            
            rollout_stats["total"] += self.args.num_rollout * \
                self.args.step_beam_size

            all_responses_first_stage = []
            all_logprobs_first_stage = []
            all_advantages_first_stage = []
            all_token_nums_first_stage = []

            # Map back to beams
            for idx, (resp, logp) in enumerate(zip(raw_responses, raw_logprobs)):
                beam_idx = idx // self.args.num_rollout
                advantage = logp - previous_values[beam_idx]

                all_responses_first_stage.append(resp)
                all_logprobs_first_stage.append(logp)
                all_advantages_first_stage.append(advantage)
                
                token_count = len(resp) // 4 # Estimate
                all_token_nums_first_stage.append(token_count)
                token_stats["output"] += token_count

            # Debug: Print responses
            print(f"\n=== First Stage Responses (Total: {len(all_responses_first_stage)}) ===")
            # for i, response in enumerate(all_responses_first_stage):
            #     print(f"Sample {i}: {response}")
            print("=" * 50)

            # ---------------------------------------------------------------
            # 2. Width Pruning
            # ---------------------------------------------------------------
            if self.args.width_pruning_strategy != "none" and self.args.width_pruning_strategy != "":
                keep_foresight_list = []
                if self.args.width_pruning_strategy == "low_sigma":
                    mean = np.mean(all_logprobs_first_stage)
                    std = np.std(all_logprobs_first_stage)
                    for idx, logp in enumerate(all_logprobs_first_stage):
                        if logp > mean - self.args.sigma_rate * std:
                            keep_foresight_list.append(idx)
                    
                    # Preserve high semantic quality responses
                    semantic_quality_indices = []
                    for idx, response in enumerate(all_responses_first_stage):
                        if idx not in keep_foresight_list:
                            semantic_score = 0
                            if len(response.strip()) > 50: semantic_score += 0.2
                            if any(ind in response.lower() for ind in ['answer:', 'therefore', 'thus', 'hence']): semantic_score += 0.1
                            if any(ind in response.lower() for ind in ['=', '+', '-', '*', '/']): semantic_score += 0.1
                            if semantic_score >= 0.3: semantic_quality_indices.append(idx)
                    
                    keep_foresight_list.extend(semantic_quality_indices)
                    keep_foresight_list = list(set(keep_foresight_list))

                if len(keep_foresight_list) < self.args.step_beam_size:
                    weights = softmax(
                        [logp/TEMPERATURE for logp in all_logprobs_first_stage])
                    num_to_add = self.args.step_beam_size - len(keep_foresight_list)
                    available_indices = [i for i in range(len(all_logprobs_first_stage)) if i not in keep_foresight_list]
                    if available_indices:
                        available_weights = [weights[i] for i in available_indices]
                        available_weights = [w/sum(available_weights) for w in available_weights]
                        additional_indices = np.random.choice(
                            available_indices,
                            size=num_to_add,
                            p=available_weights,
                            replace=False
                        ).tolist()
                        keep_foresight_list.extend(additional_indices)

                keep_foresight_list.sort()

                rollout_stats["saved"] += (self.args.step_beam_size *
                                        self.args.num_rollout - len(keep_foresight_list))

                filtered_responses = [all_responses_first_stage[i] for i in keep_foresight_list]
                filtered_logprobs = [all_logprobs_first_stage[i] for i in keep_foresight_list]
                filtered_advantages = [all_advantages_first_stage[i] for i in keep_foresight_list]
                filtered_beam_indices = [i // self.args.num_rollout for i in keep_foresight_list]

                print(f"\n=== Stage 1 Filtered (Total: {len(filtered_responses)}) ===")
                print("=" * 50)

                completion_prompts = []
                for idx in range(len(keep_foresight_list)):
                    partial  = filtered_responses[idx]
                    beam_idx = filtered_beam_indices[idx]
                    chat = self._prepare_chat_template(example, system_prompt)
                    # Inject history + partial step as the assistant prefix
                    chat[-1]["content"] = previous_steps[beam_idx] + partial
                    prompt = self.tokenizer.apply_chat_template(
                        chat, tokenize=False
                    ).rstrip(self.tokenizer.eos_token).rstrip()
                    completion_prompts.append(prompt)
                    token_stats["input"] += len(self.tokenizer(prompt)["input_ids"])

                comp_texts, comp_logprobs_raw = self._generate(
                    completion_prompts,
                    n_return=1,
                    max_new_tokens=1024,
                    stop_strs=["<end_of_reasoning>"]
                )

                rollout_stats["total"] += len(completion_prompts)

                completed_responses  = []
                completed_logprobs   = []
                completed_advantages = []

                for idx, (comp_text, comp_logp) in enumerate(zip(comp_texts, comp_logprobs_raw)):
                    beam_idx = filtered_beam_indices[idx]
                    # Full step = partial prefix + completion suffix.
                    # Stored as a unit so Token-Guard and clustering see the complete text,
                    # and next_steps is built correctly downstream.
                    full_step = (filtered_responses[idx].strip() + " " + comp_text.strip()).strip()
                    completed_responses.append(full_step)
                    completed_logprobs.append(comp_logp)
                    completed_advantages.append(comp_logp - previous_values[beam_idx])
                    token_stats["output"] += max(len(comp_text) // 4, 1)

                print(f"\n=== Stage 2 Completed (Total: {len(completed_responses)}) ===")
                print("=" * 50)
                
            beam_to_candidates = {} 
            for idx, response in enumerate(completed_responses):
                original_idx = keep_foresight_list[idx]
                parent_beam_idx = original_idx // self.args.num_rollout
                
                if parent_beam_idx not in beam_to_candidates:
                    beam_to_candidates[parent_beam_idx] = []
                beam_to_candidates[parent_beam_idx].append((idx, response))

            aligned_artifacts = [None] * len(completed_responses)
            aligned_global_scores = [0.0] * len(completed_responses)

            # 2. 批量计算
            for parent_idx, batch in beam_to_candidates.items():
                # 构造验证用 Context (System + Q + Parent History)
                chat = self._prepare_chat_template(example, system_prompt)
                chat[-1]["content"] = previous_steps[parent_idx] 
                context_text = self.tokenizer.apply_chat_template(
                    chat, tokenize=False
                ).rstrip(self.tokenizer.eos_token).rstrip()
                
                cand_indices = [item[0] for item in batch]
                cand_texts = [item[1] for item in batch]
                
                # (A) 调用插件：Token级自查 + 片段级评分 (Eq. 2-9)
                # 使用共享模型进行验证
                batch_artifacts = self.tg_scorer.verify_candidates(
                    context_text=context_text,
                    candidate_texts=cand_texts,
                    h_x=current_h_x
                )
                
                # =====================================================
                # [新增] 插入 Segment 级局部修复 (Eq. 10)
                # =====================================================
                final_batch_artifacts = []
                for art in batch_artifacts:
                    # 对每个 candidate 尝试修复
                    # 注意：这会增加推理耗时，但能显著提升质量
                    refined_art = self.tg_scorer.refine_segment(
                        context_text=context_text,
                        artifact=art,
                        h_x=current_h_x
                    )
                    final_batch_artifacts.append(refined_art)
                
                # 更新列表，后续的 Global Score 将基于修复后的片段计算
                batch_artifacts = final_batch_artifacts
                # =====================================================

                # (B) 调用插件：全局评分 (Eq. 11-14)
                parent_chain = beam_artifacts[parent_idx]
                for local_idx, artifact in enumerate(batch_artifacts):
                    # 临时链 = 历史 + 当前
                    temp_chain = parent_chain + [artifact]
                    f_global, _, _ = self.tg_scorer.compute_chain_global_score(temp_chain)
                    
                    global_idx = cand_indices[local_idx]
                    aligned_global_scores[global_idx] = f_global
                    aligned_artifacts[global_idx] = artifact

            # 3. 融合分数 (Fusion)
            # New Advantage = Old Advantage + lambda * TokenGuard_Score
            enhanced_advantages = []
            for i, adv in enumerate(completed_advantages):
                tg_bonus = self.tg_lambda * aligned_global_scores[i]
                enhanced_advantages.append(adv + tg_bonus)
                
            # [CRITICAL] 使用融合后的优势分数进行后续选择
            completed_advantages = enhanced_advantages

            # ---------------------------------------------------------------
            # 4. Third Stage: Cluster and select
            # ---------------------------------------------------------------
            try:
                clusters, cluster_info = self.cluster_and_select_chains(completed_responses, completed_advantages)
                
                if clusters is None:
                    print(f'Clustering failed: {cluster_info["reason"]}, using advantage-based')
                    weights = softmax([adv/TEMPERATURE for adv in completed_advantages])
                    selected = np.random.choice(len(weights), size=self.args.step_beam_size, p=weights, replace=False).tolist()
                    stop_foresight = False
                else:
                    cluster_list = [sorted(c) for c in clusters]
                    # Calculate weights
                    cluster_len_ratio = [len(c)/len(completed_responses) for c in cluster_list]
                    per_sample_ratio = []
                    for i in range(len(completed_responses)):
                        for c_idx, c in enumerate(cluster_list):
                            if i in c: per_sample_ratio.append(cluster_len_ratio[c_idx]); break
                    
                    cluster_weights = softmax(per_sample_ratio)
                    adv_weights = softmax([adv/TEMPERATURE for adv in completed_advantages])
                    weights = [(cluster_weights[i] + adv_weights[i])/2 for i in range(len(completed_responses))]

                    # [FIX: Insurance for Random Choice]
                    non_zero_count = np.count_nonzero(np.array(weights) > 1e-10)
                    target_size = min(len(weights), self.args.step_beam_size)
                    
                    if non_zero_count < target_size:
                        print(f"⚠️ [Insurance Triggered] Weights too sharp (only {non_zero_count} non-zeros), using Top-K.")
                        # Fallback to Top-K
                        selected = np.argsort(weights)[-target_size:].tolist()
                        selected.reverse()
                    else:
                        selected = np.random.choice(len(weights), size=target_size, p=weights, replace=False).tolist()

                    largest_ratio = max(len(c) for c in cluster_list) / len(completed_responses)
                    if largest_ratio >= self.args.threshold:
                        stop_foresight = True

                # Record info
                step_info = {
                    'first_stage': {
                        'responses': all_responses_first_stage,
                        # ... other stats ...
                    },
                    'second_stage': {
                        'responses': completed_responses,
                        'advantages': completed_advantages 
                    },
                    'final': {
                        'selected_steps': [previous_steps[keep_foresight_list[idx]//self.args.num_rollout] + completed_responses[idx] + "\n" for idx in selected],
                        'selected_indices': selected
                    }
                }
                traj_info['foresight_part'].append(step_info)

                # =================================================================
                # 5. Return with Next Artifacts (History Update)
                # =================================================================
                next_beam_artifacts = []
                next_steps_text = []
                
                for idx in selected:
                    # keep_foresight_list[idx] maps back to the original rollout index
                    original_idx = keep_foresight_list[idx]
                    parent_beam_idx = original_idx // self.args.num_rollout
                    
                    # Update history
                    parent_chain = beam_artifacts[parent_beam_idx]
                    curr_artifact = aligned_artifacts[idx]
                    
                    if curr_artifact:
                        next_beam_artifacts.append(parent_chain + [curr_artifact])
                    else:
                        next_beam_artifacts.append(parent_chain)
                        
                    full_text = previous_steps[parent_beam_idx] + completed_responses[idx] + "\n"
                    next_steps_text.append(full_text)

                return {
                    "next_steps": next_steps_text,
                    "next_values": [completed_logprobs[idx] for idx in selected],
                    "trajectories": completed_responses,
                    "steps": [keep_foresight_list[idx] for idx in selected],
                    "logprobs": completed_logprobs,
                    "advantages": completed_advantages,
                    "stop_foresight": stop_foresight,
                    "next_artifacts": next_beam_artifacts # [Token-Guard] Return artifacts
                }

            except Exception as e:
                print(f'Error in _process_step: {e}, using fallback')
                # [Fix] Ensure advantages are valid before softmax
                fallback_adv = np.nan_to_num(completed_advantages, nan=-10.0)
                
                weights = softmax([adv/TEMPERATURE for adv in fallback_adv])
                
                # Double check weights sum to 1 (floating point errors)
                weights = weights / weights.sum()
                
                # [FIX: Insurance for Random Choice in Fallback]
                non_zero_count = np.count_nonzero(weights > 1e-10)
                target_size = min(len(weights), self.args.step_beam_size)
                
                if non_zero_count < target_size:
                    print(f"⚠️ [Fallback Insurance] Using Top-K selection.")
                    selected = np.argsort(fallback_adv)[-target_size:].tolist()
                    selected.reverse()
                else:
                    selected = np.random.choice(len(weights), size=target_size, p=weights, replace=False).tolist()

                next_beam_artifacts = []
                next_steps_text = []
                for idx in selected:
                    original_idx = keep_foresight_list[idx]
                    parent_beam_idx = original_idx // self.args.num_rollout
                    
                    if idx < len(aligned_artifacts) and aligned_artifacts[idx]:
                        next_beam_artifacts.append(beam_artifacts[parent_beam_idx] + [aligned_artifacts[idx]])
                    else:
                        next_beam_artifacts.append(beam_artifacts[parent_beam_idx])
                    
                    full_text = previous_steps[parent_beam_idx] + completed_responses[idx] + "\n"
                    next_steps_text.append(full_text)

                return {
                    "next_steps": next_steps_text,
                    "next_values": [all_logprobs_first_stage[idx] for idx in selected],
                    "trajectories": all_responses_first_stage,
                    "steps": [keep_foresight_list[idx] for idx in selected],
                    "logprobs": all_logprobs_first_stage,
                    "advantages": all_advantages_first_stage,
                    "stop_foresight": stop_foresight,
                    "next_artifacts": next_beam_artifacts
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

        # Enhanced quality assessment
        if hasattr(step_results, 'advantages') and step_results['advantages']:
            avg_advantage = np.mean(step_results['advantages'])
            if avg_advantage < -2.0:
                print(f'Early stopping at depth {current_step} (poor reasoning quality, avg advantage: {avg_advantage:.3f})')
                return True

        # Check for repetitive patterns
        if len(step_results["trajectories"]) > 1:
            from difflib import SequenceMatcher
            similarity_scores = []
            for i in range(len(step_results["trajectories"])):
                for j in range(i+1, len(step_results["trajectories"])):
                    similarity = SequenceMatcher(None, 
                                               step_results["trajectories"][i], 
                                               step_results["trajectories"][j]).ratio()
                    similarity_scores.append(similarity)
            
            if similarity_scores and np.mean(similarity_scores) > 0.8:
                print(f'Early stopping at depth {current_step} (high response similarity: {np.mean(similarity_scores):.3f})')
                return True

        if self.args.depth_pruning_strategy == "cluster":
            if step_results["stop_foresight"]:
                print(
                    f'Early stopping at depth {current_step} (max cluster ratio >= args.threshold)')
                return True

        return False

    def _generate_final_response(self, example, system_prompt, previous_steps, previous_values, token_stats, rollout_stats, traj_info):
            """Generate final response after multi-step reasoning (Hybrid: API + Local)"""
            # Prepare input for each beam
            all_prompts = []
            for beam_idx in range(self.args.step_beam_size):
                chat = self._prepare_chat_template(example, system_prompt)
                chat[-1]["content"] = previous_steps[beam_idx]

                full_prompt = self.tokenizer.apply_chat_template(
                    chat,
                    tokenize=False
                ).rstrip(self.tokenizer.eos_token).rstrip()

                token_stats["input"] += len(self.tokenizer(full_prompt)["input_ids"])
                all_prompts.append(full_prompt)

            responses, logprobs = self._generate(
                all_prompts,
                n_return=1,
                max_new_tokens=3000,
                stop_strs=["<end_of_reasoning>"]
            )
            
            rollout_stats["total"] += self.args.step_beam_size

            # Collect results
            all_responses = []
            all_advantages = []
            
            for i, (resp, logp) in enumerate(zip(responses, logprobs)):
                advantage = logp - previous_values[i]
                all_responses.append(resp)
                all_advantages.append(advantage)
                # 简单估算 output token 数量
                token_stats["output"] += len(resp) // 4

            # Debug: Print final stage responses
            print(f"\n=== Final Stage Responses (Total: {len(all_responses)}) ===")
            # for i, response in enumerate(all_responses):
            #     print(f"Final {i}: {response}")
            print("=" * 50)

            # Select final response
            selected_idx = self.select_response(
                all_responses,
                logprobs,
                all_advantages
            )

            # Debug: Print final selected response
            print(f"\n=== Final Selected Response ===")
            print(f"Selected index: {selected_idx}")
            print(f"Selected response: {all_responses[selected_idx]}")
            print("=" * 50)

            # Record final results
            traj_info['final_part']['responses'] = [previous_steps[i] + all_responses[i] for i in range(len(all_responses))]
            traj_info['final_part']['responses_in_the_final_generation'] = all_responses
            traj_info['final_part']['logprobs'] = logprobs
            traj_info['final_part']['advantages'] = all_advantages
            traj_info['final_part']['selected_idx'] = selected_idx

            return {
                "response": previous_steps[selected_idx] + all_responses[selected_idx],
                "trajectories": {
                    "responses": all_responses,
                    "logprobs": logprobs,
                    "advantages": all_advantages,
                    "selected_idx": selected_idx
                }
            }
    
    def _prepare_chat_template(self, example, system_prompt):
        """
        Prepare chat template based on dataset type
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
    decoder = TokenGuardDecoder(args)

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
        
        system_prompt = decoder.get_system_prompt()
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
