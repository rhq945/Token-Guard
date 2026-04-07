"""
beam_search.py — Beam search engine for TokenGuard decoding.
Handles clustering, response selection, multi-step reasoning (process_step),
and early stopping.
"""
import numpy as np
from difflib import SequenceMatcher
from sklearn.metrics import silhouette_score
from sklearn.cluster import KMeans
from sklearn.feature_extraction.text import TfidfVectorizer

from generation_utils import softmax, TEMPERATURE


class BeamSearchEngine:
    """
    Implements the multi-step beam search loop, width/depth pruning,
    and clustering-based chain selection from PAPER_STORY.md Stage 3.
    """

    def __init__(self, args, generator, tg_scorer, prompt_builder):
        self.args = args
        self.generator = generator
        self.tg_scorer = tg_scorer
        self.prompt_builder = prompt_builder
        self.tokenizer = generator.tokenizer
        self.tg_lambda = 2.0  # fused advantage weight for TokenGuard score

    # ------------------------------------------------------------------
    # Clustering
    # ------------------------------------------------------------------

    def cluster_and_select_chains(self, responses, advantages, data_size=None):
        """
        Cluster reasoning chains using TF-IDF and KMeans.
        K=5 for large datasets (>100 samples), K=3 for small datasets (<=100).
        Falls back to self.args.cluster_num when data_size is not provided.
        (PAPER_STORY.md Stage 3: cluster_and_select_chains)
        """
        if data_size is not None:
            k = 5 if data_size > 100 else 3
        else:
            k = self.args.cluster_num  # CLI fallback

        # Filter out empty responses
        valid_indices = [i for i, r in enumerate(responses) if r.strip()]
        if len(valid_indices) < self.args.step_beam_size:
            return None, {"state": "cannot cluster", "reason": "insufficient valid responses"}

        try:
            valid_responses = [responses[i] for i in valid_indices]

            if len(set(valid_responses)) <= 1:
                return None, {"state": "cannot cluster", "reason": "all responses identical"}

            vectorizer = TfidfVectorizer(
                max_features=500,
                min_df=1,
                max_df=0.9,
                ngram_range=(1, 1),
            )
            X = vectorizer.fit_transform(valid_responses)

            if X.shape[1] < 2:
                return None, {"state": "cannot cluster", "reason": "insufficient features"}

            k = min(k, len(set(valid_responses)))

            kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
            kmeans.fit(X)

            if len(valid_responses) > 2 and len(set(kmeans.labels_)) > 1:
                silhouette_avg = silhouette_score(X, kmeans.labels_)
                if silhouette_avg < 0.05:
                    return None, {
                        "state": "cannot cluster",
                        "reason": f"poor clustering quality (silhouette: {silhouette_avg:.3f})",
                    }
            else:
                silhouette_avg = 0.0

            clusters = [[] for _ in range(k)]
            for idx, label in enumerate(kmeans.labels_):
                clusters[label].append(valid_indices[idx])

            return clusters, {
                "state": "success",
                "k": k,
                "cluster_sizes": [len(c) for c in clusters],
                "silhouette_score": silhouette_avg,
                "feature_count": X.shape[1],
            }

        except Exception as e:
            return None, {"state": "fail", "error": str(e)}

    # ------------------------------------------------------------------
    # Response selection
    # ------------------------------------------------------------------

    def select_response(self, responses, logprobs, advantages):
        """Select final response based on strategy with robustness and semantic bonus."""
        if self.args.strategy == "cluster":
            valid_indices = [idx for idx, r in enumerate(responses) if r.strip() != '']
            if len(valid_indices) == 0:
                print('all responses in the final generation are empty, use -adv no replace')
                weights = softmax([-adv / TEMPERATURE for adv in advantages])
                return np.random.choice(len(advantages), p=weights)

            if len(valid_indices) < self.args.step_beam_size:
                print('valid responses are less than step_beam_size, use adv no replace')
                weights = softmax([adv / TEMPERATURE for adv in advantages])
                return np.random.choice(len(advantages), p=weights)

            try:
                valid_responses = [responses[i] for i in valid_indices]
                valid_advantages = [advantages[i] for i in valid_indices]

                clusters, cluster_info = self.cluster_and_select_chains(
                    valid_responses, valid_advantages
                )

                if clusters is None:
                    print(
                        f"Clustering failed: {cluster_info.get('reason', 'unknown')}, "
                        "using advantage-based selection with semantic bonus"
                    )
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

                    weights = softmax([adv / TEMPERATURE for adv in enhanced_advantages])
                    selected_index_in_valid = np.random.choice(len(weights), p=weights)
                    return valid_indices[selected_index_in_valid]

                cluster_sizes = [len(c) for c in clusters]
                largest_idx = int(np.argmax(cluster_sizes))
                selected_cluster = clusters[largest_idx]

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

                weights = softmax([adv / TEMPERATURE for adv in enhanced_advantages])
                selected_index_in_cluster = np.random.choice(len(weights), p=weights)
                selected_index_in_valid = selected_cluster[selected_index_in_cluster]
                selected_index_final = valid_indices[selected_index_in_valid]

                print(
                    f'Selected from largest cluster (size: {len(selected_cluster)}) '
                    'with enhanced advantage'
                )
                return selected_index_final

            except Exception as e:
                print(
                    f'Cannot select response based on cluster: {e}, '
                    'using advantage-based selection with semantic bonus'
                )
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
                weights = softmax([adv / TEMPERATURE for adv in enhanced_advantages])
                return np.random.choice(len(weights), p=weights)

        else:
            raise ValueError(f"Unknown strategy: {self.args.strategy}")

    # ------------------------------------------------------------------
    # Multi-step reasoning
    # ------------------------------------------------------------------

    def process_step(
        self,
        example,
        system_prompt,
        previous_steps,
        previous_values,
        token_stats,
        rollout_stats,
        traj_info,
        current_h_x,
        beam_artifacts,
    ):
        """Process one reasoning step: generate → prune → complete → TokenGuard score → cluster."""
        stop_foresight = False

        # Stage 1: Generate partial responses (stopped at newline)
        all_prompts = []
        for beam_idx in range(self.args.step_beam_size):
            chat = self.prompt_builder.prepare_chat_template(example, system_prompt)
            if self.args.model_id == "mistral":
                chat[1]['content'] = system_prompt + "\n" + chat[1]['content']
                chat = chat[1:]
            base_prompt = self.tokenizer.apply_chat_template(
                chat, tokenize=False
            ).rstrip(self.tokenizer.eos_token).rstrip()
            full_prompt = base_prompt + previous_steps[beam_idx]
            all_prompts.append(full_prompt)
            token_stats["input"] += len(self.tokenizer(full_prompt)["input_ids"])

        raw_responses, raw_logprobs = self.generator.generate(
            all_prompts,
            n_return=self.args.num_rollout,
            max_new_tokens=1024,
            stop_strs=["\n", "<end_of_reasoning>"],
        )

        rollout_stats["total"] += self.args.num_rollout * self.args.step_beam_size

        all_responses_first_stage = []
        all_logprobs_first_stage = []
        all_advantages_first_stage = []
        all_token_nums_first_stage = []

        for idx, (resp, logp) in enumerate(zip(raw_responses, raw_logprobs)):
            beam_idx = idx // self.args.num_rollout
            advantage = logp - previous_values[beam_idx]

            all_responses_first_stage.append(resp)
            all_logprobs_first_stage.append(logp)
            all_advantages_first_stage.append(advantage)

            token_count = len(resp) // 4  # Estimate
            all_token_nums_first_stage.append(token_count)
            token_stats["output"] += token_count

        print(f"\n=== First Stage Responses (Total: {len(all_responses_first_stage)}) ===")
        print("=" * 50)

        # Stage 2: Width pruning then completion
        if self.args.width_pruning_strategy != "none" and self.args.width_pruning_strategy != "":
            keep_foresight_list = []
            if self.args.width_pruning_strategy == "low_sigma":
                mean = np.mean(all_logprobs_first_stage)
                std = np.std(all_logprobs_first_stage)
                for idx, logp in enumerate(all_logprobs_first_stage):
                    if logp > mean - self.args.sigma_rate * std:
                        keep_foresight_list.append(idx)

                semantic_quality_indices = []
                for idx, response in enumerate(all_responses_first_stage):
                    if idx not in keep_foresight_list:
                        semantic_score = 0
                        if len(response.strip()) > 50:
                            semantic_score += 0.2
                        if any(
                            ind in response.lower()
                            for ind in ['answer:', 'therefore', 'thus', 'hence']
                        ):
                            semantic_score += 0.1
                        if any(ind in response.lower() for ind in ['=', '+', '-', '*', '/']):
                            semantic_score += 0.1
                        if semantic_score >= 0.3:
                            semantic_quality_indices.append(idx)

                keep_foresight_list.extend(semantic_quality_indices)
                keep_foresight_list = list(set(keep_foresight_list))

            if len(keep_foresight_list) < self.args.step_beam_size:
                weights = softmax(
                    [logp / TEMPERATURE for logp in all_logprobs_first_stage]
                )
                num_to_add = self.args.step_beam_size - len(keep_foresight_list)
                available_indices = [
                    i
                    for i in range(len(all_logprobs_first_stage))
                    if i not in keep_foresight_list
                ]
                if available_indices:
                    available_weights = [weights[i] for i in available_indices]
                    available_weights = [w / sum(available_weights) for w in available_weights]
                    additional_indices = np.random.choice(
                        available_indices,
                        size=num_to_add,
                        p=available_weights,
                        replace=False,
                    ).tolist()
                    keep_foresight_list.extend(additional_indices)

            keep_foresight_list.sort()

            rollout_stats["saved"] += (
                self.args.step_beam_size * self.args.num_rollout
                - len(keep_foresight_list)
            )

            filtered_responses = [all_responses_first_stage[i] for i in keep_foresight_list]
            filtered_logprobs = [all_logprobs_first_stage[i] for i in keep_foresight_list]
            filtered_advantages = [all_advantages_first_stage[i] for i in keep_foresight_list]
            filtered_beam_indices = [i // self.args.num_rollout for i in keep_foresight_list]

            print(f"\n=== Stage 1 Filtered (Total: {len(filtered_responses)}) ===")
            print("=" * 50)

            completion_prompts = []
            for idx in range(len(keep_foresight_list)):
                partial = filtered_responses[idx]
                beam_idx = filtered_beam_indices[idx]
                chat = self.prompt_builder.prepare_chat_template(example, system_prompt)
                chat[-1]["content"] = previous_steps[beam_idx] + partial
                prompt = self.tokenizer.apply_chat_template(
                    chat, tokenize=False
                ).rstrip(self.tokenizer.eos_token).rstrip()
                completion_prompts.append(prompt)
                token_stats["input"] += len(self.tokenizer(prompt)["input_ids"])

            comp_texts, comp_logprobs_raw = self.generator.generate(
                completion_prompts,
                n_return=1,
                max_new_tokens=1024,
                stop_strs=["<end_of_reasoning>"],
            )

            rollout_stats["total"] += len(completion_prompts)

            completed_responses = []
            completed_logprobs = []
            completed_advantages = []

            for idx, (comp_text, comp_logp) in enumerate(zip(comp_texts, comp_logprobs_raw)):
                beam_idx = filtered_beam_indices[idx]
                full_step = (
                    filtered_responses[idx].strip() + " " + comp_text.strip()
                ).strip()
                completed_responses.append(full_step)
                completed_logprobs.append(comp_logp)
                completed_advantages.append(comp_logp - previous_values[beam_idx])
                token_stats["output"] += max(len(comp_text) // 4, 1)

            print(f"\n=== Stage 2 Completed (Total: {len(completed_responses)}) ===")
            print("=" * 50)

        # Group completed responses by parent beam for batched TokenGuard scoring
        beam_to_candidates = {}
        for idx, response in enumerate(completed_responses):
            parent_beam_idx = keep_foresight_list[idx] // self.args.num_rollout
            if parent_beam_idx not in beam_to_candidates:
                beam_to_candidates[parent_beam_idx] = []
            beam_to_candidates[parent_beam_idx].append((idx, response))

        aligned_artifacts = [None] * len(completed_responses)
        aligned_global_scores = [0.0] * len(completed_responses)

        for parent_idx, batch in beam_to_candidates.items():
            chat = self.prompt_builder.prepare_chat_template(example, system_prompt)
            chat[-1]["content"] = previous_steps[parent_idx]
            context_text = self.tokenizer.apply_chat_template(
                chat, tokenize=False
            ).rstrip(self.tokenizer.eos_token).rstrip()

            cand_indices = [item[0] for item in batch]
            cand_texts = [item[1] for item in batch]

            # Token-level self-check + segment scoring (Eq. 2–9)
            batch_artifacts = self.tg_scorer.verify_candidates(
                context_text=context_text,
                candidate_texts=cand_texts,
                h_x=current_h_x,
            )

            # Segment-level local refinement (Eq. 10)
            batch_artifacts = [
                self.tg_scorer.refine_segment(context_text, art, current_h_x)
                for art in batch_artifacts
            ]

            # Global chain scoring (Eq. 11–14)
            parent_chain = beam_artifacts[parent_idx]
            for local_idx, artifact in enumerate(batch_artifacts):
                temp_chain = parent_chain + [artifact]
                f_global, _, _ = self.tg_scorer.compute_chain_global_score(temp_chain)
                aligned_global_scores[cand_indices[local_idx]] = f_global
                aligned_artifacts[cand_indices[local_idx]] = artifact

        # Fuse TokenGuard score into advantage: new_adv = old_adv + λ · TG_score
        completed_advantages = [
            adv + self.tg_lambda * aligned_global_scores[i]
            for i, adv in enumerate(completed_advantages)
        ]

        # Stage 3: Cluster and select
        try:
            clusters, cluster_info = self.cluster_and_select_chains(
                completed_responses, completed_advantages
            )

            if clusters is None:
                print(f'Clustering failed: {cluster_info["reason"]}, using advantage-based')
                weights = softmax([adv / TEMPERATURE for adv in completed_advantages])
                selected = np.random.choice(
                    len(weights), size=self.args.step_beam_size, p=weights, replace=False
                ).tolist()
                stop_foresight = False
            else:
                cluster_list = [sorted(c) for c in clusters]
                cluster_len_ratio = [
                    len(c) / len(completed_responses) for c in cluster_list
                ]
                per_sample_ratio = []
                for i in range(len(completed_responses)):
                    for c_idx, c in enumerate(cluster_list):
                        if i in c:
                            per_sample_ratio.append(cluster_len_ratio[c_idx])
                            break

                cluster_weights = softmax(per_sample_ratio)
                adv_weights = softmax([adv / TEMPERATURE for adv in completed_advantages])
                weights = [
                    (cluster_weights[i] + adv_weights[i]) / 2
                    for i in range(len(completed_responses))
                ]

                non_zero_count = np.count_nonzero(np.array(weights) > 1e-10)
                target_size = min(len(weights), self.args.step_beam_size)

                if non_zero_count < target_size:
                    print(
                        f"⚠️ [Insurance Triggered] Weights too sharp "
                        f"({non_zero_count} non-zeros), using Top-K."
                    )
                    selected = np.argsort(weights)[-target_size:].tolist()
                    selected.reverse()
                else:
                    selected = np.random.choice(
                        len(weights), size=target_size, p=weights, replace=False
                    ).tolist()

                largest_ratio = (
                    max(len(c) for c in cluster_list) / len(completed_responses)
                )
                if largest_ratio >= self.args.threshold:
                    stop_foresight = True

            step_info = {
                'first_stage': {
                    'responses': all_responses_first_stage,
                },
                'second_stage': {
                    'responses': completed_responses,
                    'advantages': completed_advantages,
                },
                'final': {
                    'selected_steps': [
                        previous_steps[keep_foresight_list[idx] // self.args.num_rollout]
                        + completed_responses[idx]
                        + "\n"
                        for idx in selected
                    ],
                    'selected_indices': selected,
                },
            }
            traj_info['foresight_part'].append(step_info)

            next_beam_artifacts = []
            next_steps_text = []

            for idx in selected:
                original_idx = keep_foresight_list[idx]
                parent_beam_idx = original_idx // self.args.num_rollout
                parent_chain = beam_artifacts[parent_beam_idx]
                curr_artifact = aligned_artifacts[idx]

                if curr_artifact:
                    next_beam_artifacts.append(parent_chain + [curr_artifact])
                else:
                    next_beam_artifacts.append(parent_chain)

                full_text = (
                    previous_steps[parent_beam_idx] + completed_responses[idx] + "\n"
                )
                next_steps_text.append(full_text)

            return {
                "next_steps": next_steps_text,
                "next_values": [completed_logprobs[idx] for idx in selected],
                "trajectories": completed_responses,
                "steps": [keep_foresight_list[idx] for idx in selected],
                "logprobs": completed_logprobs,
                "advantages": completed_advantages,
                "stop_foresight": stop_foresight,
                "next_artifacts": next_beam_artifacts,
            }

        except Exception as e:
            print(f'Error in process_step: {e}, using fallback')
            fallback_adv = np.nan_to_num(completed_advantages, nan=-10.0)
            weights = softmax([adv / TEMPERATURE for adv in fallback_adv])
            weights = weights / weights.sum()
            non_zero_count = np.count_nonzero(weights > 1e-10)
            target_size = min(len(weights), self.args.step_beam_size)

            if non_zero_count < target_size:
                print("⚠️ [Fallback Insurance] Using Top-K selection.")
                selected = np.argsort(fallback_adv)[-target_size:].tolist()
                selected.reverse()
            else:
                selected = np.random.choice(
                    len(weights), size=target_size, p=weights, replace=False
                ).tolist()

            next_beam_artifacts = []
            next_steps_text = []
            for idx in selected:
                original_idx = keep_foresight_list[idx]
                parent_beam_idx = original_idx // self.args.num_rollout

                if idx < len(aligned_artifacts) and aligned_artifacts[idx]:
                    next_beam_artifacts.append(
                        beam_artifacts[parent_beam_idx] + [aligned_artifacts[idx]]
                    )
                else:
                    next_beam_artifacts.append(beam_artifacts[parent_beam_idx])

                full_text = (
                    previous_steps[parent_beam_idx] + completed_responses[idx] + "\n"
                )
                next_steps_text.append(full_text)

            return {
                "next_steps": next_steps_text,
                "next_values": [all_logprobs_first_stage[idx] for idx in selected],
                "trajectories": all_responses_first_stage,
                "steps": [keep_foresight_list[idx] for idx in selected],
                "logprobs": all_logprobs_first_stage,
                "advantages": all_advantages_first_stage,
                "stop_foresight": stop_foresight,
                "next_artifacts": next_beam_artifacts,
            }

    # ------------------------------------------------------------------
    # Early stopping
    # ------------------------------------------------------------------

    def should_stop_early(self, step_results, current_step):
        """Check if early stopping conditions are met with enhanced quality assessment."""
        if current_step < self.args.least_foresight_num:
            return False

        just_stop = True
        first_response = step_results["trajectories"][0]
        for response in step_results["trajectories"][1:]:
            if response != first_response:
                just_stop = False
                break

        if just_stop:
            print(f'Early stopping at depth {current_step} (all responses are the same)')
            return True

        if hasattr(step_results, 'advantages') and step_results['advantages']:
            avg_advantage = np.mean(step_results['advantages'])
            if avg_advantage < -2.0:
                print(
                    f'Early stopping at depth {current_step} '
                    f'(poor reasoning quality, avg advantage: {avg_advantage:.3f})'
                )
                return True

        if len(step_results["trajectories"]) > 1:
            similarity_scores = []
            for i in range(len(step_results["trajectories"])):
                for j in range(i + 1, len(step_results["trajectories"])):
                    similarity = SequenceMatcher(
                        None,
                        step_results["trajectories"][i],
                        step_results["trajectories"][j],
                    ).ratio()
                    similarity_scores.append(similarity)

            if similarity_scores and np.mean(similarity_scores) > 0.8:
                print(
                    f'Early stopping at depth {current_step} '
                    f'(high response similarity: {np.mean(similarity_scores):.3f})'
                )
                return True

        if self.args.depth_pruning_strategy == "cluster":
            if step_results["stop_foresight"]:
                print(
                    f'Early stopping at depth {current_step} '
                    '(max cluster ratio >= args.threshold)'
                )
                return True

        return False
