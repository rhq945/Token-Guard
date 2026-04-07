"""
decoder.py — TokenGuardDecoder: model loading and example processing.
Orchestrates the full decoding pipeline by composing TokenGuardGenerator,
PromptBuilder, and BeamSearchEngine.
"""
import numpy as np
import torch

from token_guard_plugin import LatentEnvironment, TokenGuardConfig
from generation_utils import TokenGuardGenerator
from prompt_builder import PromptBuilder
from beam_search import BeamSearchEngine


class TokenGuardDecoder:
    def __init__(self, args):
        self.args = args
        self.model = None
        self.tokenizer = None
        self.tg_scorer = None
        self.generator = None
        self.prompt_builder = None
        self.beam_engine = None
        self.initialize_model()

    def initialize_model(self):
        """Load the shared model via LatentEnvironment (single model, no double-loading)."""
        print(f"Loading Shared Model via TokenGuard from {self.args.model_path}...")

        tg_config = TokenGuardConfig(device="cuda")
        if getattr(self.args, 'tau_global', None) is not None:
            tg_config.tau_global = self.args.tau_global
        self.tg_scorer = LatentEnvironment(
            model_path=self.args.model_path,
            config=tg_config,
        )

        # Reuse model and tokenizer from TokenGuard (single-model design)
        self.model = self.tg_scorer.model
        self.tokenizer = self.tg_scorer.tokenizer

        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
            self.model.config.pad_token_id = self.tokenizer.eos_token_id

        print("[TokenGuard] Generator mode: Local HuggingFace")
        self.model.config.pad_token_id = self.tokenizer.eos_token_id

        np.random.seed(self.args.seed)
        torch.manual_seed(self.args.seed)

        self.generator = TokenGuardGenerator(self.model, self.tokenizer)
        self.prompt_builder = PromptBuilder(self.args)
        self.beam_engine = BeamSearchEngine(
            self.args, self.generator, self.tg_scorer, self.prompt_builder
        )

    def get_system_prompt(self, dataset_type=None):
        return self.prompt_builder.get_system_prompt(dataset_type)

    # ------------------------------------------------------------------
    # Main decoding loop
    # ------------------------------------------------------------------

    def process_example(self, example, system_prompt):
        """Process a single example through the TokenGuard decoding pipeline."""
        token_stats = {"input": 0, "output": 0}
        rollout_stats = {"total": 0, "saved": 0}

        # Stage 1 init: compute context anchor h_x and beam artifact histories
        _passage_for_anchor = self.prompt_builder.preprocess_passage(
            example['passage'], self.args.datasets, question=example.get('question', '')
        )
        init_context = (
            f"{system_prompt}\nPassage: {_passage_for_anchor}\n"
            f"Question: {example['question']}\n"
        )
        current_h_x = self.tg_scorer.initialize_anchor(init_context)
        beam_artifacts = [[] for _ in range(self.args.step_beam_size)]

        traj_pool = [[] for _ in range(self.args.num_foresight)]
        step_pool = [[] for _ in range(self.args.num_foresight)]
        prob_pool = [[] for _ in range(self.args.num_foresight + 1)]
        adv_pool = [[] for _ in range(self.args.num_foresight + 1)]

        previous_steps = ["The reasoning steps are:\n\n" for _ in range(self.args.step_beam_size)]
        previous_values = [0.0 for _ in range(self.args.step_beam_size)]

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
                'cluster_num': self.args.cluster_num,
            },
        }

        # Multi-step reasoning
        for step in range(self.args.num_foresight):
            step_results = self.beam_engine.process_step(
                example,
                system_prompt,
                previous_steps,
                previous_values,
                token_stats,
                rollout_stats,
                traj_info,
                current_h_x,
                beam_artifacts,
            )

            if self.beam_engine.should_stop_early(step_results, step):
                break

            previous_steps = step_results["next_steps"]
            previous_values = step_results["next_values"]

            if "next_artifacts" in step_results:
                beam_artifacts = step_results["next_artifacts"]
                if beam_artifacts and beam_artifacts[0]:
                    _, f_fact, f_logic = self.tg_scorer.compute_chain_global_score(
                        beam_artifacts[0]
                    )
                    self.tg_scorer.adjust_thresholds(f_fact, f_logic)

            traj_pool[step] = step_results["trajectories"]
            step_pool[step] = step_results["steps"]
            prob_pool[step] = step_results["logprobs"]
            adv_pool[step] = step_results["advantages"]

        # Global iteration: find the highest-scoring chain across all beams
        best_chain = []
        best_global_score = -1.0
        for chain in beam_artifacts:
            if chain:
                f_g, _, _ = self.tg_scorer.compute_chain_global_score(chain)
                if f_g > best_global_score:
                    best_global_score = f_g
                    best_chain = list(chain)

        # M_max=2 refinement loop targeting τ_global=0.7 (PAPER_STORY.md Stage 3)
        cannot_answer = False
        for global_iter in range(self.tg_scorer.config.m_max):
            if best_global_score >= self.tg_scorer.config.tau_global:
                print(
                    f"[GlobalIter {global_iter}] Converged: "
                    f"{best_global_score:.3f} >= τ_global={self.tg_scorer.config.tau_global:.3f}"
                )
                break

            print(
                f"[GlobalIter {global_iter}] Score {best_global_score:.3f} "
                f"< τ_global={self.tg_scorer.config.tau_global:.3f}. Refining worst segment..."
            )

            if not best_chain:
                cannot_answer = True
                break

            _, f_fact, f_logic = self.tg_scorer.compute_chain_global_score(best_chain)
            self.tg_scorer.adjust_thresholds(f_fact, f_logic)

            worst_seg_idx = min(
                range(len(best_chain)), key=lambda k: best_chain[k].segment_score
            )
            _passage_refine = self.prompt_builder.preprocess_passage(
                example['passage'], self.args.datasets, question=example.get('question', '')
            )
            init_context = (
                f"{system_prompt}\nPassage: {_passage_refine}\n"
                f"Question: {example['question']}\n"
            )
            refined_seg = self.tg_scorer.refine_segment(
                context_text=init_context,
                artifact=best_chain[worst_seg_idx],
                h_x=current_h_x,
                max_retries=1,
            )
            best_chain[worst_seg_idx] = refined_seg

            best_global_score, _, _ = self.tg_scorer.compute_chain_global_score(best_chain)

            if global_iter == self.tg_scorer.config.m_max - 1:
                if best_global_score < self.tg_scorer.config.tau_global:
                    print(
                        f"[GlobalIter] Score {best_global_score:.3f} below τ_global after "
                        f"{self.tg_scorer.config.m_max} iterations; proceeding with best chain."
                    )

        # Only output 'cannot answer' when there is literally no chain to use
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
                    "final": {},
                },
                "traj_info": traj_info,
            }

        final_result = self._generate_final_response(
            example,
            system_prompt,
            previous_steps,
            previous_values,
            token_stats,
            rollout_stats,
            traj_info,
        )

        traj_info['token_num'] = token_stats["input"] + token_stats["output"]

        return {
            "response": final_result["response"],
            "token_stats": token_stats,
            "rollout_stats": rollout_stats,
            "trajectories": {
                "steps": step_pool,
                "probs": prob_pool,
                "advantages": adv_pool,
                "final": final_result["trajectories"],
            },
            "traj_info": traj_info,
        }

    def _generate_final_response(
        self,
        example,
        system_prompt,
        previous_steps,
        previous_values,
        token_stats,
        rollout_stats,
        traj_info,
    ):
        """Generate final response from each beam, then select the best via clustering."""
        all_prompts = []
        for beam_idx in range(self.args.step_beam_size):
            chat = self.prompt_builder.prepare_chat_template(example, system_prompt)
            chat[-1]["content"] = previous_steps[beam_idx]

            full_prompt = self.tokenizer.apply_chat_template(
                chat, tokenize=False
            ).rstrip(self.tokenizer.eos_token).rstrip()

            token_stats["input"] += len(self.tokenizer(full_prompt)["input_ids"])
            all_prompts.append(full_prompt)

        responses, logprobs = self.generator.generate(
            all_prompts,
            n_return=1,
            max_new_tokens=3000,
            stop_strs=["<end_of_reasoning>"],
        )

        rollout_stats["total"] += self.args.step_beam_size

        all_responses = []
        all_advantages = []

        for i, (resp, logp) in enumerate(zip(responses, logprobs)):
            all_responses.append(resp)
            all_advantages.append(logp - previous_values[i])
            token_stats["output"] += len(resp) // 4

        print(f"\n=== Final Stage Responses (Total: {len(all_responses)}) ===")
        print("=" * 50)

        selected_idx = self.beam_engine.select_response(
            all_responses, logprobs, all_advantages
        )

        print(f"\n=== Final Selected Response ===")
        print(f"Selected index: {selected_idx}")
        print(f"Selected response: {all_responses[selected_idx]}")
        print("=" * 50)

        traj_info['final_part']['responses'] = [
            previous_steps[i] + all_responses[i] for i in range(len(all_responses))
        ]
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
                "selected_idx": selected_idx,
            },
        }
