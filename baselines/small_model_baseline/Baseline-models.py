import json
import os
import argparse
import time
import datetime
from vllm import LLM, SamplingParams
from transformers import AutoTokenizer

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_id', type=str, default='llama3.1')
    parser.add_argument('--data_path', type=str, required=True)
    parser.add_argument('--output_dir', type=str, required=True)
    parser.add_argument('--file_name', type=str, default='result')
    parser.add_argument('--time_path', type=str, default='/cpfs01/user/xufangzhi/o1/cluster_results/time/')
    parser.add_argument('--max_examples', type=int, default=300)

    args = parser.parse_args()

    if args.model_id == "llama3.1":
        model_path = "/data/rhq/halu/models/Meta-Llama-3.1-8B-Instruct"
    elif args.model_id == "qwen3":
        model_path = "/data/rhq/halu/models/qwen3-8b"
    elif args.model_id == "llama3.1-3b":
        model_path = "/data/rhq/halu/models/Llama-3.2-3B-Instruct"
    elif args.model_id == "llama-13b":
        model_path = "/data/rhq/TOKEN-GUARD/halu/models/Llama-2-13b-chat"
        PATH_TO_CONVERTED_WEIGHTS = "/data/rhq/TOKEN-GUARD/halu/models/Llama-2-13b-chat"
    else:
        raise ValueError("Unknown model_id")

    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    if not tokenizer.pad_token:
        tokenizer.pad_token = tokenizer.eos_token
    stop_token = tokenizer.eos_token

    model = LLM(model=model_path, tensor_parallel_size=1, trust_remote_code=True)

    # --- 记录开始时间 ---
    start_time = time.perf_counter()
    start_dt = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S.%f")[:-3]
    print(f"[INFO] Start processing at {start_dt}")

    with open(args.data_path) as file:
        test_data = json.load(file)

    os.makedirs(args.output_dir, exist_ok=True)
    output_path = os.path.join(args.output_dir, args.file_name + '.json')

    prompt = (
        # "Please solve the following problem step by step. "
        # "You will be presented with a question. "
        # "Answer the user's question strictly based on the given information. "
        # "Do not make up information. "
        # "At the end, output: Answer:[your answer here]."
            "You are a precise reasoning assistant. Your task is to answer questions based ONLY on the information provided in the passage.\n\n"
            "CRITICAL RULES:\n"
            "1. ONLY use information that is EXPLICITLY stated in the passage\n"
            "2. If the passage does not contain enough information to answer the question, say 'The passage does not provide enough information'\n"
            "3. Do NOT make assumptions, inferences, or add external knowledge\n"
            "4. Do NOT use phrases like 'it depends', 'possibly', 'maybe', 'likely' unless the passage explicitly indicates uncertainty\n"
            "5. Be specific and precise in your answers\n"
            "6. If you find contradictory information in the passage, acknowledge it explicitly\n\n"
            "Please solve the following problem step by step, using ONLY the given passage information.\n"
            "At the end, output: Answer:[your answer here]."
    )

    # zeroshot_map = {
    #     'default': (
    #         "You are a precise reasoning assistant. Your task is to answer questions based ONLY on the information provided in the passage.\n\n"
    #         "CRITICAL RULES:\n"
    #         "1. ONLY use information that is EXPLICITLY stated in the passage\n"
    #         "2. If the passage does not contain enough information to answer the question, say 'The passage does not provide enough information'\n"
    #         "3. Do NOT make assumptions, inferences, or add external knowledge\n"
    #         "4. Do NOT use phrases like 'it depends', 'possibly', 'maybe', 'likely' unless the passage explicitly indicates uncertainty\n"
    #         "5. Be specific and precise in your answers\n"
    #         "6. If you find contradictory information in the passage, acknowledge it explicitly\n\n"
    #         "Please solve the following problem step by step, using ONLY the given passage information.\n"
    #         "At the end, output: Answer:[your answer here]."
    #     ),
    #     'pubmedqa': (
    #         "You will be given a PubMed-style passage and a Yes/No/Maybe question.\n"
    #         "Answer rules:\n"
    #         "1. Begin with exactly one of: \"Yes.\" / \"No.\" / \"Maybe.\"\n"
    #         "2. Summarize the main conclusion from the passage in exactly ONE short sentence (≤25 words).\n"
    #         "3. Preserve key phrases and medical terms from the passage; do not replace them with synonyms.\n"
    #         "4. Always include explicitly stated conditions, subgroups, or limitations if they appear in the conclusion.\n"
    #         "5. Do NOT add recommendations, explanations, or new information.\n"
    #         "6. The final output must be exactly ONE LINE:\n"
    #         "Answer:[Yes./No./Maybe. + short sentence]"
    #     ),
    #     'financebench': (
    #         "You are an equity research analyst. Answer the question using **only the data provided**. Follow these instructions carefully:\n\n"
    #         "1. Always produce a single-line final answer.\n"
    #         "2. Do not show calculations, reasoning, or commentary.\n"
    #         "3. Match the exact format of the ground truth:\n"
    #         "   - \"$360000.00\" for USD thousands\n"
    #         "   - \"$7223.00\" for USD millions\n"
    #         "   - \"$4.90\" for USD billions\n"
    #         "   - \"34.7%\" for percentages\n"
    #         "   - \"1.08\" for ratios\n"
    #         "4. If the answer is not directly available from the statements, output:\n"
    #         "   \"Unable to answer based on given data.\"\n\n"
    #         "**Example:**\n"
    #         "Q: How much was Boeing's FY2017 interest expense (USD thousands)?\n"
    #         "A: Answer: $360000.00\n"
    #         "At the end, output: Answer:[your answer here]."
    #     ),
    #     'halueval': (
    #         "You will be presented with a question.\n"
    #         "Answer the user's question strictly based on the given information.\n"
    #         "Do not make up information.\n"
    #         "At the end, output: Answer:[your answer here]."
    #     ),
    #     'history': (
    #         "You will be presented with a question.\n"
    #         "Answer the user's question strictly based on the given information.\n"
    #         "Do not make up information.\n"
    #         "At the end, output: Answer:[your answer here]."
    #     ),
    #     'ragtruth': (
    #         "You are given passages and a question. Follow these steps:\n"
    #         "Answer the question using only the information from the given passages.\n"
    #         " - Include specific examples, numbers, or comparisons if mentioned.\n"
    #         " - Include all details that support the answer.\n"
    #         " - Do not add external information.\n"
    #         " - If the passages do not contain sufficient information, answer: \"Unable to answer based on given passages.\"\n"
    #         "At the end, output: Answer:[your answer here]"
    #     ),
    #     'covidQA': (
    #         "You will be presented with a question.\n"
    #         "Answer the user's question strictly based on the given information.\n"
    #         "Do not make up information.\n"
    #         "At the end, output: Answer:[your answer here]."
    #     )
    # }

    # # 确定 dataset_type
    # dataset_type = None
    # filename = os.path.basename(args.data_path).lower()
    # for dtype in zeroshot_map.keys():
    #     if dtype in filename:
    #         dataset_type = dtype
    #         break
    # else:
    #     dataset_type = 'default'

    # # 获取对应 prompt
    # zeroshot_prompt = zeroshot_map.get(dataset_type, zeroshot_map['default'])

    # fewshot_prompt = "Please solve the following problem step by step. You will be presented with a question. Answer the question based on the given information. At the end, output: Answer:[your answer here]."

    # # 根据 shot_mode 选择最终 system_prompt
    # prompt = zeroshot_prompt 

    with open(output_path, "w") as f:
        for i, item in enumerate(test_data[:args.max_examples]):
            passage = item.get('passage', '')
            question = item.get('question', '')
            chat = [
                {'role': 'system', 'content': prompt},
                {'role': 'user', 'content': f'Passage: {passage}\nQuestion: {question}'},
                {'role': 'assistant', 'content': ''}
            ]
            inputs = tokenizer.apply_chat_template(chat, tokenize=False).replace(stop_token, "").strip()
            sampling_params = SamplingParams(max_tokens=1024, n=1, logprobs=0, temperature=0.6, stop=["<end_of_reasoning>"])
            outputs = model.generate([inputs], sampling_params)
            response = outputs[0].outputs[0].text.strip()
            result = {
                'id': i,
                'question': question,
                'ground_truth': item.get('answer', ''),
                'passage': passage,
                'label': item.get('label', None),
                'source_ds': item.get('source_ds', None),
                'response': response
            }
            f.write(json.dumps(result, ensure_ascii=False) + '\n')
            f.flush()

    # --- 记录结束时间 ---
    end_time = time.perf_counter()
    end_dt = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S.%f")[:-3]
    elapsed_ms = (end_time - start_time) * 1000
    print(f"[INFO] Finished processing at {end_dt}")
    print(f"[INFO] Total processing time: {elapsed_ms:.2f} ms")