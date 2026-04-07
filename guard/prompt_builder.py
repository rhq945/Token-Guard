"""
prompt_builder.py — Prompt construction and passage preprocessing.
Handles dataset-specific system prompts, CovidQA passage extraction,
and chat template assembly.
"""
import re
import os

from logic_example import (
    HISTORY_8_FEW_SHOT,
    NFL_8_FEW_SHOT,
    halueval_6_FEW_SHOT,
    covidQA_4_FEW_SHOT,
    financebench_5_FEW_SHOT,
    pubmedQA_4_FEW_SHOT,
    RAGTruth_5_FEW_SHOT,
)


class PromptBuilder:
    """Builds system prompts and chat templates; pre-processes passages."""

    def __init__(self, args):
        self.args = args

    # ------------------------------------------------------------------
    # Passage preprocessing
    # ------------------------------------------------------------------

    def preprocess_passage(
        self,
        passage: str,
        dataset_type: str,
        max_chars: int = 12000,
        question: str = "",
    ) -> str:
        """
        Dataset-specific passage preprocessing.
        CovidQA passages are full academic papers with metadata headers + long Text sections.
        Strategy:
          1. Always include the FULL Abstract (no length cap) — it often contains key findings.
          2. Fill remaining budget with question-aware paragraph selection from Text body,
             always anchoring the first ~600 chars of Text (intro) + highest-scoring paragraphs.
        max_chars=12000 (≈3000 tokens) leaves enough room within max_length=4096 for
        the zeroshot system prompt (~150 tokens) + question + chat overhead (~150 tokens).
        """
        if dataset_type != 'covidQA':
            return passage

        # 1. Extract FULL Abstract (everything between 'Abstract:' and 'Text:')
        abstract = ""
        abs_idx = passage.find('Abstract:')
        text_idx = passage.find('Text:')
        if abs_idx >= 0:
            end = text_idx if (text_idx > abs_idx) else (abs_idx + 1500)
            abstract = passage[abs_idx:end].strip()

        # 2. Extract Text body
        text_body = passage[text_idx:] if text_idx >= 0 else passage

        # 3. If total content fits in budget, return it all
        total = len(abstract) + len(text_body) + 2
        if total <= max_chars:
            return (abstract + "\n\n" + text_body if abstract else text_body).strip()

        budget = max_chars - len(abstract) - 10

        if question:
            # 3a. Always keep first 600 chars of Text (intro/context)
            anchor = text_body[:600]
            remaining_budget = budget - len(anchor) - 2

            # 3b. Split rest into paragraphs, score by question keyword overlap
            question_words = set(re.findall(r'\b\w{4,}\b', question.lower()))
            rest = text_body[600:]
            paragraphs = [
                p.strip()
                for p in re.split(r'\n{2,}|\n(?=[A-Z\-\d])', rest)
                if p.strip() and len(p.strip()) > 40
            ]
            if not paragraphs:
                paragraphs = [
                    s.strip()
                    for s in re.split(r'(?<=[.!?])\s+', rest)
                    if s.strip() and len(s.strip()) > 30
                ]

            scored_idx = sorted(
                range(len(paragraphs)),
                key=lambda i: len(
                    question_words & set(re.findall(r'\b\w{4,}\b', paragraphs[i].lower()))
                ),
                reverse=True,
            )

            selected = []
            used = 0
            for i in scored_idx:
                para = paragraphs[i]
                if used + len(para) + 2 <= remaining_budget:
                    selected.append((i, para))
                    used += len(para) + 2
                if used >= remaining_budget:
                    break

            selected.sort(key=lambda x: x[0])
            relevant_text = anchor + "\n\n" + '\n\n'.join(p for _, p in selected)
        else:
            relevant_text = text_body[:budget]

        return (abstract + "\n\n" + relevant_text if abstract else relevant_text).strip()

    # ------------------------------------------------------------------
    # System prompt
    # ------------------------------------------------------------------

    def get_system_prompt(self, dataset_type=None):
        """Get the appropriate system prompt based on dataset type."""
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
                "You are a biomedical QA assistant. Answer using ONLY the provided passage.\n"
                "Rules:\n"
                "1. Copy the EXACT PHRASE or SENTENCE from the passage that answers the question.\n"
                "2. NEVER answer yes/no questions with just 'Yes.' or 'No.' — always include the specific text.\n"
                "   Wrong: 'No.'  Correct: 'the current strategy... is not sufficient because...'\n"
                "3. For list questions (viruses, methods, etc.): include ALL items from the passage.\n"
                "4. Use the FULL NAME as written (e.g., 'Human metapneumovirus (HMPV)' not 'HMPV').\n"
                "5. Do NOT paraphrase or summarize — copy exact wording.\n"
                "6. Your FINAL output line must be EXACTLY: Answer:[copied text]"
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
            "halueval": halueval_6_FEW_SHOT,
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

        # CovidQA: always force zeroshot — few-shot examples consume ~4000 tokens and
        # leave no budget for the long medical passage (avg 18k chars).
        if dataset_type == 'covidQA':
            return zeroshot_map.get('covidQA', default_zeroshot)

        if self.args.shot_mode == 'zeroshot':
            return zeroshot_map.get(dataset_type, default_zeroshot)

        else:  # fewshot
            base_prompt = zeroshot_map.get(dataset_type)
            if base_prompt is None:
                base_prompt = default_zeroshot

            fewshot_examples = fewshot_map.get(dataset_type, "")
            if not fewshot_examples:
                return base_prompt

            system_prompt = (
                f"{base_prompt}\n\n"
                "I will give you some examples for reference:\n"
                f"{fewshot_examples}"
            )
            return system_prompt

    # ------------------------------------------------------------------
    # Chat template
    # ------------------------------------------------------------------

    def prepare_chat_template(self, example, system_prompt):
        """
        Prepare chat template based on dataset type.
        For CovidQA, passage is preprocessed to strip metadata and use question-aware extraction.
        """
        question = example['question']
        passage = self.preprocess_passage(
            example['passage'], self.args.datasets, question=question
        )
        chat = [
            {'role': 'system', 'content': system_prompt},
            {
                'role': 'user',
                'content': (
                    'Passage: ' + passage + '\nQuestion: ' + question
                    + '\nPlease directly follow the previous reasoning steps (if provided) and generate the remaining ones.\n'
                ),
            },
            {'role': 'assistant', 'content': ''},
        ]
        return chat
