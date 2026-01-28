import json
import random
import numpy as np
import torch
import os
import argparse
import time
import datetime
# from data.math_example import MATH_POT_FEW_SHOT, MATH_COT_FEW_SHOT, GSM_COT_8_SHOT, MATH_COT_4_SHOT
# from data.logic_example import LOGIC_MRC_COT_4_SHOT
from vllm import LLM, SamplingParams
from transformers import (
    StoppingCriteriaList,
    StoppingCriteria,
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig
)
import re
INF = 10

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--datasets', type=str, default='gsm')
    parser.add_argument('--model_id', type=str, default='llama3.1')
    parser.add_argument('--data_path', type=str, default='/cpfs01/user/xufangzhi/o1/data/reclor_val.json')
    parser.add_argument('--output_dir', type=str, default='/cpfs01/user/xufangzhi/o1/cluster_results/')
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--step_beam_size', type=int, default=4)
    parser.add_argument('--num_rollout', type=int, default=4)  # sc baseline用 num rollout来进行控制
    parser.add_argument('--num_foresight', type=int, default=4)
    parser.add_argument('--record_process', type=bool, default=False)
    parser.add_argument('--strategy', type=str, default='test')
    parser.add_argument('--time_path', type=str, default='/cpfs01/user/xufangzhi/o1/cluster_results/time/')
    parser.add_argument('--file_name', type=str, default='test')
    parser.add_argument('--max_examples', type=int, default=50)
    parser.add_argument('--shot_mode', type=str, default='zeroshot', choices=['zeroshot', 'fewshot'],
                        help='选择zeroshot或fewshot模式')
    args = parser.parse_args()
    ffname = args.file_name
    args.output_path = args.output_dir + ffname  + '.json'
    ffname = args.file_name

    if args.model_id=="llama3.1":
        PATH_TO_CONVERTED_WEIGHTS = "/data/rhq/TOKEN-GUARD/halu/models/Meta-Llama-3.1-8B-Instruct"
    elif args.model_id=="qwen3":
        PATH_TO_CONVERTED_WEIGHTS = "/data/rhq/TOKEN-GUARD/halu/models/qwen3-8b"
    # elif args.model_id=="mistral":
    #     PATH_TO_CONVERTED_WEIGHTS = "/nas/shared/NLP_A100/hf_hub/models--mistralai--Mistral-7B-Instruct-v0.3/snapshots/83e9aa141f2e28c82232fea5325f54edf17c43de"
    # elif args.model_id=="gemma":
    #     PATH_TO_CONVERTED_WEIGHTS = "/nas/shared/NLP_A100/hf_hub/models--google--gemma-2-9b-it/snapshots/11c9b309abf73637e4b6f9a3fa1e92e615547819"
    tokenizer = AutoTokenizer.from_pretrained(PATH_TO_CONVERTED_WEIGHTS, trust_remote_code=True)
        # tokenizer = AutoTokenizer.from_pretrained(PATH_TO_CONVERTED_WEIGHTS, max_length=2048, trust_remote_code=True)

    if not tokenizer.pad_token:
        tokenizer.pad_token = tokenizer.eos_token
    stop_token = tokenizer.eos_token

    start_time = time.time()
    model = LLM(model=PATH_TO_CONVERTED_WEIGHTS, tensor_parallel_size=1, trust_remote_code=True)
    # model = LLM(model=PATH_TO_CONVERTED_WEIGHTS, tensor_parallel_size=1, trust_remote_code=True, max_model_len=4096)

            # --- 记录开始时间 ---
    start_time = time.perf_counter()
    start_dt = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S.%f")[:-3]
    print(f"[INFO] Start processing at {start_dt}")

    num_rollout = args.num_rollout
    num_foresight = args.num_foresight
    step_beam_size = args.step_beam_size
    
    DATA_PATH = args.data_path
    with open(DATA_PATH) as file:
        test_data = json.load(file)

    OUTPUT_PATH = args.output_path
    # zeroshot_prompt = (
    # # "Please solve the following problem step by step. "
    # # "You will be presented with a question. "
    # # "Answer the user's question strictly based on the given information. "
    # # "Do not make up information. "
    # # "At the end, output: Answer:[your answer here]."
    #             "You are a precise reasoning assistant. Your task is to answer questions based ONLY on the information provided in the passage.\n\n"
    #             "CRITICAL RULES:\n"
    #         "1. ONLY use information that is EXPLICITLY stated in the passage\n"
    #         "2. If the passage does not contain enough information to answer the question, say 'The passage does not provide enough information'\n"
    #         "3. Do NOT make assumptions, inferences, or add external knowledge\n"
    #         "4. Do NOT use phrases like 'it depends', 'possibly', 'maybe', 'likely' unless the passage explicitly indicates uncertainty\n"
    #         "5. Be specific and precise in your answers\n"
    #         "6. If you find contradictory information in the passage, acknowledge it explicitly\n\n"
    #         "Please solve the following problem step by step, using ONLY the given passage information.\n"
    #         "At the end, output: Answer:[your answer here]."
    # )
    # fewshot_prompt = "Please solve the following problem step by step. You will be presented with a question. Answer the question based on the given information. At the end, output: Answer:[your answer here]."

    # if args.shot_mode == 'zeroshot':
    #     system_prompt = zeroshot_prompt
    # else:
    #     system_prompt = fewshot_prompt 
    zeroshot_map = {
        'default': (
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
        ),
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
        )
    }

    # 确定 dataset_type
    dataset_type = None
    filename = os.path.basename(args.data_path).lower()
    for dtype in zeroshot_map.keys():
        if dtype in filename:
            dataset_type = dtype
            break
    else:
        dataset_type = 'default'

    # 获取对应 prompt
    zeroshot_prompt = zeroshot_map.get(dataset_type, zeroshot_map['default'])

    fewshot_prompt = "Please solve the following problem step by step. You will be presented with a question. Answer the question based on the given information. At the end, output: Answer:[your answer here]."

    # 根据 shot_mode 选择最终 system_prompt
    system_prompt = zeroshot_prompt if args.shot_mode == 'zeroshot' else fewshot_prompt

    iadx = 0
    all_output_token_num = 0
    all_input_token_num = 0

    max_num = len(test_data) if args.max_examples == -1 else min(len(test_data), args.max_examples)
    with open(OUTPUT_PATH, "w") as f:
        for i in range(max_num):
            # if i > 3:
            #     break
            try_time = 0
            while try_time < 3:
                try:
                    # 对于每一个问题
                    # if iadx == 1:
                    #     break
                    problem_start_time = time.time()
                    output_token_num_for_this_question = 0
                    input_token_num_for_this_question = 0
                    iadx += 1
                    result = {}
                    result['id'] = i
                    result['response'] = ''
                    all_res = []

                    passage = test_data[i]['passage']
                    question = test_data[i]['question']
                    chat = [
                        {'role': 'system', 'content': system_prompt},
                        {'role': 'user', 'content': 'Passage: ' + passage + '\nQuestion: ' + question + '\nPlease directly follow the previous reasoning steps (if provided) and generate the remaining ones.\n'},
                        {'role': 'assistant', 'content': ''}
                    ]

                    if args.model_id=="mistral" or args.model_id=="gemma":
                        chat[1]['content']= system_prompt +"\n" + chat[1]['content']
                        chat =chat[1:]

                    inputs = tokenizer.apply_chat_template(
                        chat,
                        tokenize=False, # 输出的是 str，而不是token_ids
                    )
                    # print(inputs)
                    inputs = inputs.replace(stop_token, "").strip()
                    
                    inputs_list = [inputs] # 对每个step beam size进行

                    for each_input in inputs_list:
                        input_token_num_for_this_question += len(tokenizer(each_input)['input_ids'])


                    sampling_params = SamplingParams(max_tokens=1024, n=num_rollout, logprobs=0, temperature=0.6, stop=["<end_of_reasoning>"])

                    outputs = model.generate(inputs_list, sampling_params)

                    for _ in range(num_rollout):
                        output = outputs[0].outputs[_]
                        response = output.text.strip()
                        output_token_num_for_this_question += len(output.token_ids)
                        all_res.append(response)
                    break
                except:
                    try_time += 1
                    if try_time == 3:
                        with open('/cpfs01/user/xufangzhi/o1/cluster_results/over_length_results/'+ffname+'.txt', 'a') as fsa:
                            fsa.write(str(i))
                            fsa.write('\n')
            all_input_token_num += input_token_num_for_this_question
            all_output_token_num += output_token_num_for_this_question
            print(f"question {i} input token num: {input_token_num_for_this_question}")
            print(f"question {i} output token num: {output_token_num_for_this_question}")
            print(f"all input token num: {all_input_token_num}")
            print(f"all output token num: {all_output_token_num}")

            result = {}
            result['id'] = i
            result['question'] = test_data[i]['question']
            result['ground_truth'] = test_data[i]['answer']
            result['passage'] = test_data[i]['passage']
            result['label'] = test_data[i].get('label', None)
            result['source_ds'] = test_data[i].get('source_ds', None)

            result['response_all_beams'] = all_res
            f.write(json.dumps(result) + '\n')
            f.flush() 
        
        end_time = time.perf_counter()
        end_dt = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S.%f")[:-3]
        elapsed_ms = (end_time - start_time) * 1000
        print(f"[INFO] Finished processing at {end_dt}")
        print(f"[INFO] Total processing time: {elapsed_ms:.2f} ms")

        end_time = time.time()
        time_span = end_time - start_time
        print(f"time: {time_span}")
        time_path = args.time_path + ffname + '.txt'
        with open(time_path, 'a') as f:
            f.write('time:  ' + str(time_span) + '\n')
            f.write('num_rollout:  ' + str(num_rollout) + '\n')
            f.write('num_foresight:  ' + str(num_foresight) + '\n')
            f.write('all_output_token_num:  ' + str(all_output_token_num) + '\n')
            f.write('all_input_token_num:  ' + str(all_input_token_num) + '\n')
