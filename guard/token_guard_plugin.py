import math
import torch
import torch.nn.functional as F
import numpy as np
from dataclasses import dataclass, field
from typing import List, Dict, Tuple, Optional
from transformers import AutoModelForCausalLM, AutoTokenizer


@dataclass
class TokenGuardConfig:
    # Token-level hyperparameters (PAPER_STORY.md Stage 1)
    lambda_val: float = 0.6       # Eq. 3
    tau_token: float = 0.4        # Eq. 4 — propagation gate

    # Segment-level hyperparameters (PAPER_STORY.md Stage 2)
    alpha: float = 0.5            # Eq. 9
    beta: float = 0.3             # Eq. 9
    gamma: float = 0.2            # Eq. 9
    tau_seg_low: float = 0.55
    tau_seg_high: float = 0.75

    # Global hyperparameters (PAPER_STORY.md Stage 3)
    tau_global: float = 0.7        # Eq. 13 — global convergence threshold (PAPER_STORY.md Stage 3)
    delta_tau: float = 0.1        # Eq. 15 — dynamic threshold margin
    m_max: int = 2                 # Max global iterations

    device: str = "cuda" if torch.cuda.is_available() else "cpu"


@dataclass
class CandidateSegment:
    """Segment-level intermediate artifact storing vectors for global scoring."""
    text: str
    H_k: torch.Tensor             # Aggregated segment representation vector
    e_tilde_k: torch.Tensor       # Mean input embedding
    f_norm: float                 # Token score vector norm
    segment_score: float          # F_halu^seg
    token_scores: List[float]
    evidence_score: float = 1.0


@dataclass
class RefinementAdvice:
    needed: bool
    window: Optional[Tuple[int, int]] = None
    target_token_idx: int = -1


class LatentEnvironment:
    def __init__(self, model_path: str, config: TokenGuardConfig = TokenGuardConfig()):
        self.config = config
        self.device = config.device

        print(f"[TokenGuard-GPU] Loading model on {self.device}: {model_path}")

        self.tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)

        # Load model in FP16, pinned to a single GPU to avoid cross-device tensor errors
        self.model = AutoModelForCausalLM.from_pretrained(
            model_path,
            trust_remote_code=True,
            torch_dtype=torch.float16,
            device_map={"": self.device}
        ).eval()

        # Resize embeddings if tokenizer vocab is larger (e.g. Llama3 / Qwen)
        if len(self.tokenizer) > self.model.config.vocab_size:
            self.model.resize_token_embeddings(len(self.tokenizer))

        self.current_threshold_adj = 0.0

    def _get_model_outputs(self, input_ids):
        """Forward pass returning logits, second-to-last hidden states, and input embeddings."""
        with torch.no_grad():
            outputs = self.model(input_ids, output_hidden_states=True)
        hidden_states = outputs.hidden_states[-2]    # second-to-last layer for semantics
        input_embeddings = outputs.hidden_states[0]  # embedding layer for context anchor
        return outputs.logits, hidden_states, input_embeddings

    def initialize_anchor(self, context_text: str) -> torch.Tensor:
        """[Eq. 1] Compute initial context anchor h_x = mean hidden state of input."""
        inputs = self.tokenizer(
            context_text,
            return_tensors="pt",
            truncation=True,
            max_length=4096
        ).to(self.device)
        _, hiddens, _ = self._get_model_outputs(inputs.input_ids)
        return torch.mean(hiddens[0], dim=0)

    # ------------------------------------------------------------------
    # Named scoring methods — strict alignment with PAPER_STORY.md
    # ------------------------------------------------------------------

    def hybrid_token_level_scoring(self, sim: float, p_t: float) -> float:
        """[Eq. 3] Hybrid token-level score: λ·sim + (1−λ)·p_t  (λ=0.6)."""
        return self.config.lambda_val * sim + (1 - self.config.lambda_val) * p_t

    def calculate_weighted_token_score(
        self,
        token_scores_t: torch.Tensor,
        seg_hiddens: torch.Tensor,
    ) -> Tuple[float, torch.Tensor]:
        """
        Aggregates token reliability using softmax weights w_i.
        Returns (score_token, H_k): scalar weighted score and aggregated
        segment representation vector H_k.
        """
        weights = F.softmax(token_scores_t, dim=0)
        H_k = torch.sum(weights.unsqueeze(1) * seg_hiddens, dim=0)
        score_token = torch.sum(weights * token_scores_t).item()
        return score_token, H_k

    def calculate_local_consistency(self, seg_hiddens: torch.Tensor) -> float:
        """Computes semantic transition smoothness via mean cosine similarity of consecutive
        normalized hidden states. Raw L2 norm is not used because LLM hidden states are
        unnormalized (norms >> 1), which would always yield zero with the naive formula."""
        if seg_hiddens.shape[0] > 1:
            h_norm = F.normalize(seg_hiddens, p=2, dim=-1)
            cos_sims = (h_norm[:-1] * h_norm[1:]).sum(dim=-1)  # [N-1]
            return max(0.0, cos_sims.mean().item())
        return 1.0

    def calculate_global_alignment(
        self, H_k: torch.Tensor, h_x: torch.Tensor
    ) -> float:
        """Cosine similarity of aggregated segment vector H_k against context anchor h_x."""
        return F.cosine_similarity(H_k, h_x, dim=0).item()

    def segment_level_hallucination_score(
        self, score_token: float, score_const: float, score_align: float
    ) -> float:
        """[Eq. 9] α·score_token + β·score_const + γ·score_align (α=0.5, β=0.3, γ=0.2)."""
        return (
            self.config.alpha * score_token
            + self.config.beta * score_const
            + self.config.gamma * score_align
        )

    def calculate_factual_consistency(self, chain: List[CandidateSegment]) -> float:
        """Computes F_fact(R): norm-weighted average of per-segment hallucination scores."""
        numerators = [seg.f_norm * seg.evidence_score for seg in chain]
        denom = sum(numerators) + 1e-8
        return sum(
            (numerators[k] / denom) * chain[k].segment_score
            for k in range(len(chain))
        )

    def calculate_logical_coherence(self, chain: List[CandidateSegment]) -> float:
        """
        Computes F_logic(R): λ_k · cos(H_k, H_{k+1}) averaged over consecutive pairs.
        λ_k = (1 + cos(e_k, e_{k+1})) / 2  is the embedding-level contextual factor.
        """
        K = len(chain)
        if K < 2:
            return 1.0
        logic_scores = []
        for k in range(K - 1):
            H_k    = chain[k].H_k.to(self.device)
            H_next = chain[k + 1].H_k.to(self.device)
            e_k    = chain[k].e_tilde_k.to(self.device)
            e_next = chain[k + 1].e_tilde_k.to(self.device)
            cos_hidden = F.cosine_similarity(H_k, H_next, dim=0).item()
            cos_embed  = F.cosine_similarity(e_k, e_next, dim=0).item()
            lambda_k   = (1.0 + cos_embed) / 2.0      # contextual factor λ_k
            logic_scores.append(lambda_k * cos_hidden)
        return sum(logic_scores) / (K - 1)

    # ------------------------------------------------------------------

    def verify_candidates(self, context_text: str, candidate_texts: List[str], h_x: torch.Tensor) -> List[CandidateSegment]:
        """[Eq. 2–9] Token-level self-check and segment-level scoring for each candidate."""
        if not candidate_texts:
            return []

        results = []

        if h_x.device.type != self.device:
            h_x = h_x.to(self.device)

        ctx_inputs = self.tokenizer(
            context_text,
            return_tensors="pt",
            truncation=True,
            max_length=8192  # match full_text max_length so ctx boundary is accurate
        )
        ctx_len = ctx_inputs.input_ids.shape[1]

        for text in candidate_texts:
            if not text.strip():
                continue

            full_text = context_text + text
            inputs = self.tokenizer(
                full_text,
                return_tensors="pt",
                truncation=True,
                max_length=8192
            ).to(self.device)

            input_ids = inputs.input_ids
            current_ctx_len = min(ctx_len, input_ids.shape[1] - 1)
            if current_ctx_len >= input_ids.shape[1]:
                continue

            logits, full_hiddens, full_embeds = self._get_model_outputs(input_ids)

            seg_logits = logits[0, current_ctx_len-1:-1, :]
            seg_ids = input_ids[0, current_ctx_len:]

            log_probs = F.log_softmax(seg_logits, dim=-1)
            token_log_probs = log_probs.gather(1, seg_ids.unsqueeze(1)).squeeze(1)
            token_probs = torch.exp(token_log_probs)

            seg_hiddens = full_hiddens[0, current_ctx_len:, :]
            seg_embeds  = full_embeds[0, current_ctx_len:, :]

            # Token-level scoring loop with τ_token=0.4 propagation gate (Eq. 4)
            token_scores = []
            running_h_mean = h_x.clone()
            t_offset = 1

            for i in range(len(seg_hiddens)):
                h_t = seg_hiddens[i]
                p_t = token_probs[i].item()
                sim = F.cosine_similarity(h_t, running_h_mean, dim=0).item()
                f_token = self.hybrid_token_level_scoring(sim, p_t)
                token_scores.append(f_token)

                # Only tokens above τ_token update the running semantic state
                if f_token >= self.config.tau_token:
                    running_h_mean = (running_h_mean * t_offset + h_t) / (t_offset + 1)
                    t_offset += 1

            token_scores_t = torch.tensor(token_scores, device=self.device)

            # Segment representation and scoring
            score_token, H_k = self.calculate_weighted_token_score(token_scores_t, seg_hiddens)
            e_tilde_k = torch.mean(seg_embeds, dim=0)
            f_norm = torch.norm(token_scores_t).item()

            score_const = self.calculate_local_consistency(seg_hiddens)
            score_align = self.calculate_global_alignment(H_k, h_x)
            f_seg = self.segment_level_hallucination_score(score_token, score_const, score_align)

            # Store vectors on CPU to avoid VRAM accumulation during long reasoning chains
            results.append(CandidateSegment(
                text=text,
                H_k=H_k.cpu(),
                e_tilde_k=e_tilde_k.cpu(),
                f_norm=f_norm,
                segment_score=f_seg,
                token_scores=token_scores
            ))

        return results

    def compute_chain_global_score(self, chain: List[CandidateSegment]) -> Tuple[float, float, float]:
        """
        [global_iteration_correction] Soft-min combination of F_fact and F_logic.
        Soft-Min: softmin(a, b; τ) = -τ · log(exp(-a/τ) + exp(-b/τ)), τ=0.1
        """
        if not chain:
            return 0.0, 0.0, 0.0

        f_fact  = self.calculate_factual_consistency(chain)
        f_logic = self.calculate_logical_coherence(chain)

        _tau_sm = 0.1
        _a = max(f_fact, 1e-8)
        _b = max(f_logic, 1e-8)
        f_global = -_tau_sm * math.log(
            math.exp(-_a / _tau_sm) + math.exp(-_b / _tau_sm)
        )
        f_global = max(0.0, min(1.0, f_global))

        return f_global, f_fact, f_logic

    def check_refinement(self, artifact: CandidateSegment) -> RefinementAdvice:
        """[Section 3.3] Determine if local refinement is needed based on segment score thresholds."""
        adj_low  = self.config.tau_seg_low  - self.current_threshold_adj
        adj_high = self.config.tau_seg_high + self.current_threshold_adj

        if adj_low <= artifact.segment_score < adj_high:
            if not artifact.token_scores:
                return RefinementAdvice(False)
            min_idx = np.argmin(artifact.token_scores)
            start = max(0, min_idx - 1)
            end   = min(len(artifact.token_scores), min_idx + 2)
            return RefinementAdvice(True, (start, end), min_idx)

        return RefinementAdvice(False)

    def refine_segment(self, context_text: str, artifact: CandidateSegment, h_x: torch.Tensor, max_retries: int = 3) -> CandidateSegment:
        """[Eq. 10] Local refinement via self-correction prompts at the weakest token position."""
        advice = self.check_refinement(artifact)
        if not advice.needed or max_retries <= 0:
            return artifact

        best_artifact    = artifact
        current_artifact = artifact

        # Self-correction triggers injected at the cutoff point
        correction_triggers = [
            " Wait, let me double check the details.",
            " Actually, looking closer at the context,",
            " To be precise,"
        ]

        for i in range(max_retries):
            tokens_scores = current_artifact.token_scores
            if not tokens_scores:
                break

            min_idx    = np.argmin(tokens_scores)
            cutoff_idx = max(0, min_idx - 2)
            seg_input_ids = self.tokenizer.encode(current_artifact.text, add_special_tokens=False)
            if cutoff_idx >= len(seg_input_ids):
                cutoff_idx = max(0, len(seg_input_ids) - 1)

            prefix_ids  = seg_input_ids[:cutoff_idx]
            prefix_text = self.tokenizer.decode(prefix_ids, skip_special_tokens=True)

            trigger           = correction_triggers[i % len(correction_triggers)]
            guided_input_text = context_text + prefix_text + trigger

            inputs = self.tokenizer(
                guided_input_text,
                return_tensors="pt",
                truncation=True,
                max_length=4096
            ).to(self.device)

            with torch.no_grad():
                outputs = self.model.generate(
                    **inputs,
                    max_new_tokens=len(seg_input_ids) - cutoff_idx + 30,
                    do_sample=True,
                    temperature=0.7,
                    top_p=0.9,
                    pad_token_id=self.tokenizer.pad_token_id,
                    eos_token_id=self.tokenizer.eos_token_id
                )

            full_decoded = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
            if trigger in full_decoded:
                new_suffix = full_decoded.split(trigger)[-1].strip()
            else:
                input_len  = inputs.input_ids.shape[1]
                new_suffix = self.tokenizer.decode(outputs[0][input_len:], skip_special_tokens=True)

            new_segment_text = prefix_text + " " + new_suffix

            refined_artifacts = self.verify_candidates(context_text, [new_segment_text], h_x)
            if not refined_artifacts:
                break
            refined_artifact = refined_artifacts[0]

            if refined_artifact.segment_score > best_artifact.segment_score:
                best_artifact    = refined_artifact
                current_artifact = refined_artifact
                adj_high = self.config.tau_seg_high + self.current_threshold_adj
                if best_artifact.segment_score >= adj_high:
                    break

        return best_artifact

    def adjust_thresholds(self, f_fact: float, f_logic: float):
        """[Eq. 15] Dynamic threshold adjustment based on Fact/Logic score divergence (Δτ=0.1)."""
        if f_fact < 0.5 and f_logic > 0.6:
            self.current_threshold_adj = self.config.delta_tau
            print("[TokenGuard] Thresholds TIGHTENED (Low Factuality).")
        elif f_logic < 0.5 and f_fact > 0.6:
            self.current_threshold_adj = -self.config.delta_tau
            print("[TokenGuard] Thresholds RELAXED (Low Logic).")
        else:
            self.current_threshold_adj = 0.0
