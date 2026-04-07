"""
generation_utils.py — Low-level generation helpers.
Provides softmax, TEMPERATURE constant, and TokenGuardGenerator (wraps HF model.generate).
"""
import numpy as np
import torch
from typing import List, Tuple

TEMPERATURE = 0.3  # Softmax sampling temperature


def softmax(x):
    x = np.array(x)
    x = np.nan_to_num(x, nan=-1e9, posinf=1e9, neginf=-1e9)
    x_max = np.max(x)
    if x_max == -1e9:  # all inputs were NaN/Inf
        return np.ones(len(x)) / len(x)
    e_x = np.exp(x - x_max)
    return e_x / e_x.sum(axis=0)


class TokenGuardGenerator:
    """Wraps a HuggingFace causal-LM model for batched text generation."""

    def __init__(self, model, tokenizer):
        self.model = model
        self.tokenizer = tokenizer

    def generate(
        self,
        prompts: List[str],
        n_return: int,
        max_new_tokens: int = 1024,
        stop_strs: List[str] = None,
    ) -> Tuple[List[str], List[float]]:
        """Generate text using the shared HF model. Falls back to greedy on sampling errors."""
        responses = []
        cumulative_logprobs = []

        if self.tokenizer.pad_token_id is None:
            self.tokenizer.pad_token_id = self.tokenizer.eos_token_id
            self.model.config.pad_token_id = self.tokenizer.eos_token_id

        for prompt in prompts:
            start_count = len(responses)

            inputs = self.tokenizer(
                prompt,
                return_tensors="pt",
                truncation=True,
                max_length=4096,
            ).to(self.model.device)

            input_len = inputs.input_ids.shape[1]
            outputs = None

            with torch.no_grad():
                try:
                    outputs = self.model.generate(
                        **inputs,
                        max_new_tokens=max_new_tokens,
                        do_sample=True,
                        temperature=0.7,
                        top_p=0.95,
                        top_k=50,
                        repetition_penalty=1.0,
                        num_return_sequences=n_return,
                        return_dict_in_generate=True,
                        output_scores=True,
                        pad_token_id=self.tokenizer.pad_token_id,
                        eos_token_id=self.tokenizer.eos_token_id,
                    )
                except RuntimeError as e:
                    print(f"⚠️ Generation Error: {e}. Falling back to greedy decoding.")
                    outputs = self.model.generate(
                        **inputs,
                        max_new_tokens=max_new_tokens,
                        do_sample=False,
                        num_return_sequences=1,
                        return_dict_in_generate=True,
                        output_scores=True,
                        pad_token_id=self.tokenizer.pad_token_id,
                        eos_token_id=self.tokenizer.eos_token_id,
                    )
                    if n_return > 1:
                        outputs.sequences = outputs.sequences.repeat(n_return, 1)
                        if outputs.scores:
                            outputs.scores = tuple(
                                s.repeat(n_return, 1) for s in outputs.scores
                            )

            if outputs is not None and outputs.scores:
                transition_scores = self.model.compute_transition_scores(
                    outputs.sequences, outputs.scores, normalize_logits=True
                )
            else:
                transition_scores = torch.zeros((n_return, 1)).to(self.model.device)

            for i in range(outputs.sequences.shape[0]):
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
                    avg_logprob = (
                        transition_scores[i, :valid_len].sum().item() / (valid_len + 1e-8)
                    )
                else:
                    avg_logprob = -1.0
                cumulative_logprobs.append(avg_logprob)

            # Pad to n_return if fewer sequences were returned
            expected_total = start_count + n_return
            while len(responses) < expected_total:
                responses.append(responses[-1] if len(responses) > start_count else "")
                cumulative_logprobs.append(-1.0)

        return responses, cumulative_logprobs
