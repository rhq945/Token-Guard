# cluster成两类，然后取数量较多的那一类，之后进行adv no replace
import os
os.environ["MKL_SERVICE_FORCE_INTEL"] = "1"
import json
import random
import numpy as np
import torch
import time
import datetime

import argparse
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
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

INF = 10
os.environ["CUDA_VISIBLE_DEVICES"] = "1"
import time
temp=0.1
def softmax(x):
    e_x = np.exp(np.array(x))
    return e_x / e_x.sum(axis=0)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--max_examples', type=int, default=50)
    # parser.add_argument('--time_path', type=str, default='/cpfs01/user/xufangzhi/o1/cluster_results/time/')
    parser.add_argument('--shot_mode', type=str, default='zeroshot', choices=['zeroshot', 'fewshot'],
                        help='选择zeroshot或fewshot模式')
    parser.add_argument('--datasets', type=str, default='gsm')
    parser.add_argument('--model_id', type=str, default='llama3.1')
    parser.add_argument('--data_path', type=str, default='/cpfs01/user/xufangzhi/o1/data/math_500_test.json')
    parser.add_argument('--output_dir', type=str, default='/cpfs01/user/xufangzhi/o1/cluster_results/')
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--step_beam_size', type=int, default=1)
    parser.add_argument('--num_rollout', type=int, default=2)
    parser.add_argument('--num_foresight', type=int, default=2)
    parser.add_argument('--record_process', type=bool, default=True)
    parser.add_argument('--strategy', type=str, default='sir_no_replace') # adv_no_replace, none, cluster
    parser.add_argument('--time_path', type=str, default='/cpfs01/user/xufangzhi/o1/cluster_results/time/')
    parser.add_argument('--final_select_strategy', type=str, default='same_as_strategy')
    parser.add_argument('--pruning_strategy', type=str, default='none') # none, sirno_low_sigma, advno_low_sigma, advno_above_avg, adv
    parser.add_argument('--sigma_rate', type=float, default=1.0)
    parser.add_argument('--depth_pruning_strategy', type=str, default='none') # early stopping [around_mean, within_sigma]
    parser.add_argument('--threshold', type=float, default=0.95) # threshold for early stopping
    parser.add_argument('--adv_above0', type=str, default='True')
    parser.add_argument('--file_name', type=str, default='test')
    parser.add_argument('--cluster_num', type=int, default=3)
    parser.add_argument('--gpus', type=int, default=1)
    args = parser.parse_args()
    # device = "cuda:3"
    start_time = time.time()

    ffname = args.file_name
    args.output_path = args.output_dir + ffname + '.json'

    if args.model_id=="llama3.1":
        PATH_TO_CONVERTED_WEIGHTS = "/data/rhq/halu/models/Meta-Llama-3.1-8B-Instruct"
    elif args.model_id=="qwen3":
        PATH_TO_CONVERTED_WEIGHTS = "/data/rhq/halu/models/qwen3-8b"
    elif args.model_id == "llama3.1-3b":
        model_path = "/data/rhq/halu/models/Llama-3.2-3B-Instruct"
        PATH_TO_CONVERTED_WEIGHTS = "/data/rhq/halu/models/Llama-3.2-3B-Instruct"      
    elif args.model_id == "llama-13b":
        model_path = "/data/rhq/TOKEN-GUARD/halu/models/Llama-2-13b-chat"
        PATH_TO_CONVERTED_WEIGHTS = "/data/rhq/TOKEN-GUARD/halu/models/Llama-2-13b-chat"
    # elif args.model_id=="mistral":
    #     PATH_TO_CONVERTED_WEIGHTS = "/nas/shared/NLP_A100/hf_hub/models--mistralai--Mistral-7B-Instruct-v0.3/snapshots/83e9aa141f2e28c82232fea5325f54edf17c43de"
    # elif args.model_id=="gemma":
    #     PATH_TO_CONVERTED_WEIGHTS = "/nas/shared/NLP_A100/hf_hub/models--google--gemma-2-9b-it/snapshots/11c9b309abf73637e4b6f9a3fa1e92e615547819"
    tokenizer = AutoTokenizer.from_pretrained(PATH_TO_CONVERTED_WEIGHTS, trust_remote_code=True)
    # tokenizer = AutoTokenizer.from_pretrained(PATH_TO_CONVERTED_WEIGHTS, max_length=8192, trust_remote_code=True)

    if not tokenizer.pad_token:
        tokenizer.pad_token = tokenizer.eos_token
    stop_token = tokenizer.eos_token
    print('**********************\n')
    print(stop_token)
    print('\n**********************\n')
    # exit(0)
    saved_rollout_times = 0
    total_rollout_times = 0
    model = LLM(model=PATH_TO_CONVERTED_WEIGHTS, tensor_parallel_size=args.gpus, trust_remote_code=True)

    num_rollout = args.num_rollout
    num_foresight = args.num_foresight
    step_beam_size = args.step_beam_size
    
    DATA_PATH = args.data_path
    with open(DATA_PATH) as file:
        test_data = json.load(file)

    max_num = len(test_data) if args.max_examples == -1 else min(len(test_data), args.max_examples)
    

    all_input_token_num = 0
    all_output_token_num = 0
    OUTPUT_PATH = args.output_path
    # zeroshot_prompt = (
    #     # "Please solve the following problem step by step. "
    #     # "You will be presented with a question. "
    #     # "Answer the user's question strictly based on the given information. "
    #     # "Do not make up information. "
    #     # "At the end, output: Answer:[your answer here]."
    #                 "You are a precise reasoning assistant. Your task is to answer questions based ONLY on the information provided in the passage.\n\n"
    #                 "CRITICAL RULES:\n"
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
    import time
    with open(OUTPUT_PATH, "w") as f:
        print('len test data', len(test_data))
        
        all_traj_info = []
        try_time = 0
        for i in range(max_num):
            while try_time < 3:
                # try:
                    # if i <= 436:
                    #     continue
                    # if iadx == 1:
                    #     break
                    iadx += 1
                    # print(iadx)
                    # 对于每一个问题
                    traj_pool = [[] for _ in range(num_foresight)]
                    step_pool = [[] for _ in range(num_foresight)]
                    prob_pool = [[] for _ in range(num_foresight+1)]
                    adv_pool = [[] for _ in range(num_foresight+1)]

                    traj_info = {'question_idx':i, 'passage':test_data[i]['passage'], 'question':test_data[i]['question'], 'ground_truth':test_data[i]['answer'], 'foresight_part':[], 'final_part':{}}
                    all_token_num = 0 # 这个也是用来统计输出token数量的
                    input_token_num_for_this_question = 0
                    output_token_num_for_this_question = 0

                    # 记录下来，用于最后一步的选择
                    all_cumulative_logprob_list = [[] for _ in range(step_beam_size)]
                    token_num_list = [[] for _ in range(step_beam_size)]
                    
                    sample_id_pool = []
                    
                    traj_complete = False
                    previous_steps_list = ["The reasoning steps are:\n\n" for _ in range(step_beam_size)]
                    previous_q_value_list = [0.0 for _ in range(step_beam_size)]
                    each_step_prob_pool = [[] for _ in range(step_beam_size)]
                    T = 0
                    for T in range(num_foresight):
                        # 对每个问题foresight次数，达到foresight次数后，剩下的直接补全
                        reasoning_steps_list = previous_steps_list

                        cur_foresight_info = {'foresight_epoch':T, 'steps':[], 'foresight_steps':[], 'cluster_info':{}} # 用于记录traj信息

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
                        # print('--------------------------\n')
                        # print("inputs:\n", inputs,)
                        # print('\n--------------------------\n')
                        # exit(0)
                        inputs = inputs.replace(stop_token, "").strip()
                        
                        inputs_list = [inputs + reasoning_steps_list[beam_idx] for beam_idx in range(step_beam_size)] # 对每个step beam size进行
                        # 统计一下输入的token数量
                        for each_input in inputs_list: 
                            input_token_num_for_this_question += len(tokenizer(each_input)['input_ids'])
                        # print(inputs_list[0])
                        sampling_params = SamplingParams(max_tokens=1024, n=num_rollout, logprobs=0, temperature=0.6, stop=["\n", "<end_of_reasoning>"])
                        # 根据当前的状态，foresight
                        outputs = model.generate(inputs_list, sampling_params)
                        total_rollout_times += num_rollout*step_beam_size

                        # 如果rollout的step的prob都很高且方差不大，就可以不用foresight，直接随机采
                        all_avg_logp = []
                        selected_steps = []
                        inputs_list = []
                        candidates_list = []
                        normalized_logp_list = []
                        cumulative_logprob_list = []
                        output_token_num_list = []
                        aaa = [] # 这个变量只有下面那一个用处，就是帮助记录选择的那一步所用的logprob，ctrl f 找一下那一步就懂了
                        directly_sample_flag = False
                        cur_step_token_length = []
                        cur_step_cumulative_logprob = []
                        for ii in range(step_beam_size):
                            for jj in range(num_rollout):
                                output = outputs[ii].outputs[jj]
                                response = output.text.strip()
                                cur_step_token_length.append(len(output.token_ids))
                                cur_step_cumulative_logprob.append(output.cumulative_logprob)
                                selected_steps.append(response)
                                candidates_list.append(response)
                                all_avg_logp.append(output.cumulative_logprob / (len(output.token_ids)+1e-8))
                                aaa.append(output.cumulative_logprob / (len(output.token_ids)+1e-8))
                                cumulative_logprob_list.append(output.cumulative_logprob)
                                output_token_num_list.append(len(output.token_ids))
                                all_token_num += len(output.token_ids)
                                # 给traj info记录信息
                                tem_step_info = {'index':ii*num_rollout+jj, 'text':response, 
                                                'adv':output.cumulative_logprob / (len(output.token_ids)+1e-8) - previous_q_value_list[ii]
                                                ,'logp':output.cumulative_logprob / (len(output.token_ids)+1e-8),
                                                'cumulative_logp':output.cumulative_logprob ,'token_num':len(output.token_ids)}
                                cur_foresight_info['steps'].append(tem_step_info)
                                output_token_num_for_this_question += len(output.token_ids)
                        
                        all_avg_logp = np.array(all_avg_logp)

                        # 选出需要继续foresight的idx
                        keep_foresight_list = []
                        # print('alalala')
                        # input('')

                        if args.pruning_strategy == "none" or args.pruning_strategy == "":
                            keep_foresight_list = list(range(step_beam_size*num_rollout))
                        else:
                            normalized_logp_list = []
                            adv_list = []
                            not_foresight_list = []
                            num_to_foresight = 0
                            not_foresight_normalized_logp_list = []
                            not_foresight_adv_list = []
                            for beam_idx in range(step_beam_size):
                                for j in range(num_rollout):
                                    output = outputs[beam_idx].outputs[j]
                                    normalized_logp_list.append(output.cumulative_logprob / (len(output.token_ids)+1e-8))
                                    adv_list.append(output.cumulative_logprob / (len(output.token_ids)+1e-8) - previous_q_value_list[beam_idx])
                            
                            # 计算normalized_logp_list的均值
                            if args.pruning_strategy == "sirno_low_sigma":
                                mean = np.mean(normalized_logp_list)
                                std = np.std(normalized_logp_list)

                                for iidx, each_logp in enumerate(normalized_logp_list):
                                    if each_logp > mean - args.sigma_rate * std:
                                        keep_foresight_list.append(iidx)
                                        num_to_foresight += 1
                                    else:
                                        not_foresight_list.append(iidx)
                                        not_foresight_normalized_logp_list.append(each_logp)
                                
                                if num_to_foresight < step_beam_size:
                                    # 需要补充至少够step_beam_size个
                                    temp = 0.1
                                    num_to_add = step_beam_size - num_to_foresight

                                    weights = softmax([logp/temp for logp in not_foresight_normalized_logp_list])
                                    added_from_not_to_foresight_list = np.random.choice(len(weights), p=weights, size=num_to_add, replace=False).tolist()
                                    add_idx = [not_foresight_list[i] for i in added_from_not_to_foresight_list]
                                    keep_foresight_list += add_idx
                                    keep_foresight_list.sort()
                                    
                            elif args.pruning_strategy == "sirno_high_sigma":
                                mean = np.mean(normalized_logp_list)
                                std = np.std(normalized_logp_list)

                                for iidx, each_logp in enumerate(normalized_logp_list):
                                    if each_logp > mean + args.sigma_rate * std:
                                        keep_foresight_list.append(iidx)
                                        num_to_foresight += 1
                                    else:
                                        not_foresight_list.append(iidx)
                                        not_foresight_normalized_logp_list.append(each_logp)
                                
                                if num_to_foresight < step_beam_size:
                                    # 需要补充至少够step_beam_size个
                                    temp = 0.1
                                    num_to_add = step_beam_size - num_to_foresight

                                    weights = softmax([logp/temp for logp in not_foresight_normalized_logp_list])
                                    added_from_not_to_foresight_list = np.random.choice(len(weights), p=weights, size=num_to_add, replace=False).tolist()
                                    add_idx = [not_foresight_list[i] for i in added_from_not_to_foresight_list]
                                    keep_foresight_list += add_idx
                                    keep_foresight_list.sort()
                            elif args.pruning_strategy == "sirno_both_sigma":
                                mean = np.mean(normalized_logp_list)
                                std = np.std(normalized_logp_list)

                                for iidx, each_logp in enumerate(normalized_logp_list):
                                    if abs(each_logp - mean) < args.sigma_rate * std:
                                        keep_foresight_list.append(iidx)
                                        num_to_foresight += 1
                                    else:
                                        not_foresight_list.append(iidx)
                                        not_foresight_normalized_logp_list.append(each_logp)
                                
                                if num_to_foresight < step_beam_size:
                                    # 需要补充至少够step_beam_size个
                                    temp = 0.1
                                    num_to_add = step_beam_size - num_to_foresight

                                    weights = softmax([logp/temp for logp in not_foresight_normalized_logp_list])
                                    added_from_not_to_foresight_list = np.random.choice(len(weights), p=weights, size=num_to_add, replace=False).tolist()
                                    add_idx = [not_foresight_list[i] for i in added_from_not_to_foresight_list]
                                    keep_foresight_list += add_idx
                                    keep_foresight_list.sort()

                            elif args.pruning_strategy == 'advno_low_sigma':
                                mean = np.mean(adv_list)
                                std = np.std(adv_list)

                                for iidx, each_adv in enumerate(adv_list):
                                    if each_adv > mean - args.sigma_rate * std:
                                        keep_foresight_list.append(iidx)
                                        num_to_foresight += 1
                                    else:
                                        not_foresight_list.append(iidx)
                                        not_foresight_adv_list.append(each_adv)

                                if num_to_foresight < step_beam_size:
                                    # 需要补充至少够step_beam_size个
                                    temp = 0.1
                                    num_to_add = step_beam_size - num_to_foresight
                                    
                                    weights = softmax([logp/temp for logp in not_foresight_adv_list])
                                    added_from_not_to_foresight_list = np.random.choice(len(weights), p=weights, size=num_to_add, replace=False).tolist()
                                    add_idx = [not_foresight_list[i] for i in added_from_not_to_foresight_list]
                                    keep_foresight_list += add_idx
                                    keep_foresight_list.sort()

                            elif args.pruning_strategy == 'advno_above_avg':
                                for iidx, each_adv in enumerate(adv_list):
                                    if each_adv > 0:
                                        keep_foresight_list.append(iidx)
                                        num_to_foresight += 1
                                    else:
                                        not_foresight_list.append(iidx)
                                        not_foresight_adv_list.append(each_adv)

                                if num_to_foresight < step_beam_size:
                                    # 需要补充至少够step_beam_size个
                                    temp = 0.1
                                    num_to_add = step_beam_size - num_to_foresight

                                    weights = softmax([logp/temp for logp in not_foresight_adv_list])
                                    added_from_not_to_foresight_list = np.random.choice(len(weights), p=weights, size=num_to_add, replace=False).tolist()
                                    add_idx = [not_foresight_list[i] for i in added_from_not_to_foresight_list]
                                    keep_foresight_list += add_idx
                                    keep_foresight_list.sort()

                            elif args.pruning_strategy == 'advno_high_sigma':
                                mean = np.mean(adv_list)
                                std = np.std(adv_list)

                                for iidx, each_adv in enumerate(adv_list):
                                    if each_adv > mean + args.sigma_rate * std:
                                        keep_foresight_list.append(iidx)
                                        num_to_foresight += 1
                                    else:
                                        not_foresight_list.append(iidx)
                                        not_foresight_adv_list.append(each_adv)

                                if num_to_foresight < step_beam_size:
                                    # 需要补充至少够step_beam_size个
                                    temp = 0.1
                                    num_to_add = step_beam_size - num_to_foresight

                                    weights = softmax([logp/temp for logp in not_foresight_adv_list])
                                    added_from_not_to_foresight_list = np.random.choice(len(weights), p=weights, size=num_to_add, replace=False).tolist()
                                    add_idx = [not_foresight_list[i] for i in added_from_not_to_foresight_list]
                                    keep_foresight_list += add_idx
                                    keep_foresight_list.sort()

                            elif args.pruning_strategy == 'advno_both_sigma':
                                mean = np.mean(adv_list)
                                std = np.std(adv_list)

                                for iidx, each_adv in enumerate(adv_list):
                                    if abs(each_adv-mean) < args.sigma_rate * std:
                                        keep_foresight_list.append(iidx)
                                        num_to_foresight += 1
                                    else:
                                        not_foresight_list.append(iidx)
                                        not_foresight_adv_list.append(each_adv)

                                if num_to_foresight < step_beam_size:
                                    # 需要补充至少够step_beam_size个
                                    temp = 0.1
                                    num_to_add = step_beam_size - num_to_foresight

                                    weights = softmax([logp/temp for logp in not_foresight_adv_list])
                                    added_from_not_to_foresight_list = np.random.choice(len(weights), p=weights, size=num_to_add, replace=False).tolist()
                                    add_idx = [not_foresight_list[i] for i in added_from_not_to_foresight_list]
                                    keep_foresight_list += add_idx
                                    keep_foresight_list.sort()
                        
                                
                        print('本次foresight ', len(keep_foresight_list))
                        keep_foresight_list.sort()
                        total_rollout_times += len(keep_foresight_list) # 这里加的是后面foresight的次数
                        saved_rollout_times += (step_beam_size*num_rollout - len(keep_foresight_list))
                        inputs_list = []
                        candidates_list = ['' for _ in range(len(keep_foresight_list))]  
                        for idx, foresight_idx in enumerate(keep_foresight_list):
                            output = outputs[foresight_idx//num_rollout].outputs[foresight_idx%num_rollout]
                            response = output.text.strip()
                            reasoning_steps_candidate = reasoning_steps_list[foresight_idx//num_rollout] + "\n" + response
                            candidates_list[idx] = response

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
                                
                            inputs_list.append(tokenizer.apply_chat_template(
                                chat,
                                tokenize=False,
                            ).rstrip(stop_token).rstrip() + reasoning_steps_candidate)
                            
                        # 把每一步进行补全
                        sampling_params = SamplingParams(max_tokens=1024 ,n=1, logprobs=1, stop="<end_of_reasoning>")
                        for each_input in inputs_list: 
                            input_token_num_for_this_question += len(tokenizer(each_input)['input_ids'])
                        outputs = model.generate(inputs_list, sampling_params)

                        # 用于选择
                        normalized_logp_list = []
                        advantages_list = []
                        output_text_list = []
                        foresight_token_num_list = []
                        foresight_cumulative_logp_list = []
                        for jf in range(len(inputs_list)):
                            output = outputs[jf].outputs[0]
                            response = output.text.strip()
                            normalized_logp_list.append(output.cumulative_logprob / (len(output.token_ids)+1e-8))
                            advantages_list.append(output.cumulative_logprob / (len(output.token_ids)+1e-8) - previous_q_value_list[keep_foresight_list[jf]//num_rollout])
                            output_text_list.append(response)
                            all_token_num += len(output.token_ids)
                            foresight_token_num_list.append(len(output.token_ids))
                            foresight_cumulative_logp_list.append(output.cumulative_logprob)

                            # 给traj info记录信息
                            tem_foresight_info = {'index':jf, 'idx_in_origin':keep_foresight_list[jf], 'text':response,
                                                'adv':output.cumulative_logprob / (len(output.token_ids)+1e-8) - previous_q_value_list[keep_foresight_list[jf]//num_rollout],
                                                'logp':output.cumulative_logprob / (len(output.token_ids)+1e-8),
                                                'cumulative_logp':output.cumulative_logprob ,'token_num':len(output.token_ids)}
                            cur_foresight_info['foresight_steps'].append(tem_foresight_info)
                            output_token_num_for_this_question += len(output.token_ids)

                        # 通过不同策略选择index
                        if args.strategy == "highest":  # select the step with highest rewards
                            selected_index = normalized_logp_list.index(max(normalized_logp_list))
                        elif args.strategy == "random":  # select the step with random rewards
                            selected_index = random.randint(0, num_rollout-1)
                        elif args.strategy == "sir":
                            temp = 0.1
                            def softmax(x):
                                e_x = np.exp(np.array(x))
                                return e_x / e_x.sum(axis=0)
                            weights = softmax([logp/temp for logp in normalized_logp_list])
                            selected_index_list = np.random.choice(len(weights), p=weights, size=step_beam_size).tolist()
                        elif args.strategy == "sir_no_replace":
                            temp = 0.1
                            all_length = []
                            all_cumulative_logps = []
                            normalized_logp_list = []

                            for iop in range(len(cur_step_token_length)):
                                all_length.append(cur_step_token_length[iop] + foresight_token_num_list[iop])
                                all_cumulative_logps.append(cur_step_cumulative_logprob[iop] + foresight_cumulative_logp_list[iop])
                                normalized_logp_list.append(all_cumulative_logps[iop] / (all_length[iop]+1e-8))
                            # print('@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@')
                            # input('')
                            # print('len all_length: ', len(all_length))
                            # print('len normalized_logp_list: ', len(normalized_logp_list))
                            def softmax(x):
                                e_x = np.exp(np.array(x))
                                return e_x / e_x.sum(axis=0)
                            weights = softmax([logp/temp for logp in normalized_logp_list])
                            selected_index_list = np.random.choice(len(weights), p=weights, size=step_beam_size, replace=False).tolist()
                        elif args.strategy == "adv":
                            temp = 0.1
                            def softmax(x):
                                e_x = np.exp(np.array(x))
                                return e_x / e_x.sum(axis=0)
                            
                            weights = softmax([logp/temp for logp in advantages_list])
                            selected_index_list = np.random.choice(len(weights), p=weights, size=step_beam_size).tolist()
                        elif args.strategy == "adv_no_replace":
                            temp = 0.1
                            def softmax(x):
                                e_x = np.exp(np.array(x))
                                return e_x / e_x.sum(axis=0)
                            
                            weights = softmax([logp/temp for logp in advantages_list])
                            selected_index_list = np.random.choice(len(weights), p=weights, size=step_beam_size, replace=False).tolist()
                        
                        elif args.strategy == 'cluster':
                            # mask 掉adv小于0的step,同时记录其在原来的list中的index。如果数量小于step_beam_size，就随机补充
                            tem_output_text_list = []
                            tem_selected_index_list = []
                            tem_cluster_info = {}
                            if args.adv_above0 == "True":
                                mask = [adv > 0 for adv in advantages_list] # len(advantages_list) == len（keep_foresight_list）
                                for ddi in range(len(mask)): # ddi没有具体意义，就代表下标
                                    if mask[ddi] and output_text_list[ddi] != '': # 要求adv大于0并且text不为空
                                        tem_selected_index_list.append(ddi)
                                tem_selected_index_list.sort()
                            else:
                                # tem_selected_index_list = list(range(len(advantages_list))) 
                                for dadagad in range(len(output_text_list)): # 去掉空字符串
                                    if output_text_list[dadagad] != '':
                                        tem_selected_index_list.append(dadagad)

                            
                            if len(tem_selected_index_list) < step_beam_size:
                                tem_cluster_info['state'] = 'cannot cluster'
                                print('cannot cluster, use adv no replace')
                                weights = softmax([adv/temp for adv in advantages_list])
                                selected_index_list = np.random.choice(len(weights), p=weights, size=step_beam_size, replace=False).tolist()
                            else:
                                try:
                                    # len(output_text_list) == len(advantages_list) == len(keep_foresight_list)
                                    tem_output_text_list = [output_text_list[ddi] for ddi in tem_selected_index_list] # tem_selected_index_list中元素是在keep_foresight_list中的下标
                                    tem_advantages_list = [advantages_list[ddi] for ddi in tem_selected_index_list]
                                    vectorizer =TfidfVectorizer()
                                    X=vectorizer.fit_transform(tem_output_text_list)
                                    k = args.cluster_num # TODO:后面可能会改
                                    kmeans = KMeans(n_clusters=k)
                                    kmeans.fit(X)
                                    cluster_labels = kmeans.labels_
                                    cluster_list = [[] for _ in range(k)]

                                    for aidx, cluster_label in enumerate(cluster_labels):
                                        cluster_list[cluster_label].append(aidx)
                                    cluster_list = [sorted(cluster) for cluster in cluster_list]

                                    cluster_len_ratio = [len(cluster)/len(tem_selected_index_list) for cluster in cluster_list]
                                    per_sample_cluster_len_ratio = [cluster_len_ratio[cluster_labels[ddi]] for ddi in range(len(tem_selected_index_list))]
                                    cluster_weights = softmax(per_sample_cluster_len_ratio)
                                    adv_weights = softmax([adv/temp for adv in tem_advantages_list])
                                    weights = [cluster_weights[ddi] + adv_weights[ddi] for ddi in range(len(tem_selected_index_list))]
                                    weights = [each_weight/2 for each_weight in weights]
                                    selected_index_list = np.random.choice(len(weights), p=weights, size=step_beam_size, replace=False).tolist()
                                    # idx_in_tem = [ddi for ddi in selected_index_list]
                                    selected_index_list = [tem_selected_index_list[ddi] for ddi in selected_index_list]
                                    
                                    # 用于记录 traj
                                    cluster_list_foresight_idx = [[] for _ in range(k)] # 在keep_foresight_list中的下标
                                    for aidx, cluster_label in enumerate(cluster_labels):
                                        cluster_list_foresight_idx[cluster_label].append(tem_selected_index_list[aidx])
                                    cluster_list_origin_idx = [[keep_foresight_list[foresight_idx] for foresight_idx in fcluster] for fcluster in cluster_list_foresight_idx]
                                    tem_cluster_info['state'] = 'success'
                                    tem_cluster_info['cluster_result_cluster_idx'] = cluster_list
                                    tem_cluster_info['cluster_result_foresight_idx'] = cluster_list_foresight_idx
                                    tem_cluster_info['cluster_result_origin_idx'] = cluster_list_origin_idx
                                    

                                except:
                                    tem_cluster_info['state'] =  'fail'
                                    print('cannot cluster, use adv no replace')
                                    weights = softmax([adv/temp for adv in advantages_list])
                                    selected_index_list = np.random.choice(len(weights), p=weights, size=step_beam_size, replace=False).tolist()
                            cur_foresight_info['cluster_info'] = tem_cluster_info
                            
                        # 用于记录 traj info
                        cur_foresight_info['selected_idx_in_foresight'] = selected_index_list
                        cur_foresight_info['selected_idx_in_origin'] = [keep_foresight_list[iiiidx] for iiiidx in selected_index_list]

                        traj_info['foresight_part'].append(cur_foresight_info)

                        # sample_id_pool.append([keep_foresight_list[iiiidx] for iiiidx in selected_index_list]) # 选出来的idx指的是在keep_foresight_list中的idx 
                        # 给each_step_prob_pool加东西
                        tem_logprob_list = [[] for _ in range(step_beam_size)]
                        for iii in range(step_beam_size):
                            tem_logprob_list[iii] = all_cumulative_logprob_list[iii]

                                        
                        
                        previous_steps_list_updated, previous_q_value_list = [], []
                        for m, selected_index in enumerate(selected_index_list):
                            previous_steps_list_updated.append(previous_steps_list[keep_foresight_list[selected_index]//num_rollout] + candidates_list[selected_index].strip() + "\n") # 注意这里的candidates_list长度不是num_rollout*step_beam_size
                            previous_q_value_list.append(normalized_logp_list[selected_index])

                        previous_steps_list = previous_steps_list_updated

                        selected_cumulative_logprob_list = [] # 用以判断是否还要继续foresight, 存的都是本次foresight的数据
                        idx_in_origin_list = [keep_foresight_list[iiia] for iiia in selected_index_list]
                        for jjjj in range(step_beam_size):
                            each_step_prob_pool[jjjj].append(aaa[idx_in_origin_list[jjjj]])
                            all_cumulative_logprob_list[jjjj] = tem_logprob_list[idx_in_origin_list[jjjj]//num_rollout] + [cumulative_logprob_list[idx_in_origin_list[jjjj]]]      
                            token_num_list[jjjj] = token_num_list[idx_in_origin_list[jjjj]//num_rollout] + [output_token_num_list[idx_in_origin_list[jjjj]]]   
                            selected_cumulative_logprob_list.append(cumulative_logprob_list[idx_in_origin_list[jjjj]])       

                        if args.depth_pruning_strategy != "none" and args.depth_pruning_strategy != "":

                            stop_foresight = False
                            if args.depth_pruning_strategy == "around_mean":
                                threshold = float(args.threshold)
                                mean = np.mean(selected_cumulative_logprob_list)
                                # 如果所有logp均在mean附近，就停止
                                if all(abs(each_logp - mean) < abs((1-threshold)*mean) for each_logp in selected_cumulative_logprob_list):
                                    stop_foresight = True

                            elif args.depth_pruning_strategy == "within_sigma":
                                threshold = float(args.threshold)
                                mean = np.mean(selected_cumulative_logprob_list)
                                std = np.std(selected_cumulative_logprob_list)
                                # 如果在一个标准差内的logp数量和总数的比例大于threshold，就停止
                                if len([each_logp for each_logp in selected_cumulative_logprob_list if abs(each_logp - mean) < std])/len(selected_cumulative_logprob_list) > threshold:
                                    stop_foresight = True

                            if stop_foresight and T >= 1: # 至少foresight 2次
                                break
                    
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
                        tokenize=False,
                        add_generation_prompt=True
                    ).rstrip(stop_token).rstrip()

                    inputs_list = [inputs + previous_steps_list[beam_idx] for beam_idx in range(step_beam_size)]
                    sampling_params = SamplingParams(max_tokens=3000 ,n=step_beam_size, logprobs=0, stop="<end_of_reasoning>")
                    for each_input in inputs_list: 
                        input_token_num_for_this_question += len(tokenizer(each_input)['input_ids'])
                    outputs = model.generate(inputs_list, sampling_params)
                    total_rollout_times += step_beam_size
                    
                    candidates_list = []
                    normalized_logp_list = []
                    advantages_list = []
                    answer_avg_logprob_list = []
                    output_text_list = []
                    final_answer_info = []
                    for jx in range(step_beam_size):
                        output = outputs[jx].outputs[0]
                        response = output.text.strip()
                        candidates_list.append(response)
                        normalized_logp_list.append(output.cumulative_logprob / (len(output.token_ids)+1e-8))
                        advantages_list.append(output.cumulative_logprob / (len(output.token_ids)+1e-8) - previous_q_value_list[jx])
                        prob_pool[T+1].append(output.cumulative_logprob / (len(output.token_ids)+1e-8))
                        adv_pool[T+1].append(output.cumulative_logprob / (len(output.token_ids)+1e-8) - previous_q_value_list[jx])
                        all_cumulative_logprob_list[jx].append(output.cumulative_logprob)
                        token_num_list[jx].append(len(output.token_ids))
                        output_text_list.append(response)
                        all_token_num += len(output.token_ids)
                        # 给traj info记录信息
                        tem_answer_info = {'index':jx, 'text':response, 'adv':output.cumulative_logprob / (len(output.token_ids)+1e-8) - previous_q_value_list[jx],
                                        'logp':output.cumulative_logprob / (len(output.token_ids)+1e-8),
                                        'cumulative_logp':output.cumulative_logprob ,'token_num':len(output.token_ids)}
                        final_answer_info.append(tem_answer_info)
                        output_token_num_for_this_question += len(output.token_ids)
                    traj_info['final_part']['final_answer'] = final_answer_info
                    print(output_token_num_for_this_question)
                    # 此时就只选一个index了
                    # args.strategy = args.final_strategy
                    if args.final_select_strategy == "same_as_strategy":
                        if args.strategy == "cluster":
                            # mask 掉adv小于0的step
                            tem_output_text_list = []
                            tem_selected_index_list = []
                            cur_cluster_info = {}
                            if args.adv_above0 == "True":
                                mask = [adv > 0 for adv in advantages_list] # len(advantages_list) == len（keep_foresight_list）
                                for ddi in range(len(mask)): # ddi没有具体意义，就代表下标
                                    if mask[ddi] and output_text_list[ddi] != '': # 要求adv大于0并且text不为空
                                        tem_selected_index_list.append(ddi)
                            else:
                                # tem_selected_index_list = list(range(len(advantages_list))) 
                                for dadagad in range(len(output_text_list)): # 去掉空字符串
                                    if output_text_list[dadagad] != '':
                                        tem_selected_index_list.append(dadagad)

                            if len(tem_selected_index_list) == 0:
                                # 利用adv no replace
                                cur_cluster_info['state'] = 'fail'
                                weights = softmax([-adv/temp for adv in advantages_list]) #TODO:这里加了个负号
                                selected_index_final = np.random.choice(len(weights), p=weights)
                            else:
                                try:
                                    # 注释掉的这一部分是P1+P2的方法
                                    """tem_output_text_list = [output_text_list[ddi] for ddi in tem_selected_index_list] # tem_selected_index_list中元素是在keep_foresight_list中的下标
                                    tem_advantages_list = [advantages_list[ddi] for ddi in tem_selected_index_list]
                                    vectorizer =TfidfVectorizer()
                                    X=vectorizer.fit_transform(tem_output_text_list)
                                    k = args.cluster_num # TODO:后面可能会改
                                    kmeans = KMeans(n_clusters=k)
                                    kmeans.fit(X)
                                    cluster_labels = kmeans.labels_
                                    cluster_list = [[] for _ in range(k)]

                                    for aidx, cluster_label in enumerate(cluster_labels):
                                        cluster_list[cluster_label].append(aidx)

                                    cluster_len_ratio = [len(cluster)/len(tem_selected_index_list) for cluster in cluster_list]
                                    per_sample_cluster_len_ratio = [cluster_len_ratio[cluster_labels[ddi]] for ddi in range(len(tem_selected_index_list))]
                                    cluster_weights = softmax(per_sample_cluster_len_ratio)
                                    adv_weights = softmax([adv/temp for adv in tem_advantages_list])
                                    weights = [cluster_weights[ddi] + adv_weights[ddi] for ddi in range(len(tem_selected_index_list))]
                                    weights = [each_weight/2 for each_weight in weights]
                                    selected_index_final = np.random.choice(len(weights), p=weights)
                                    selected_index_final = tem_selected_index_list[selected_index_final]"""
                                    # selected_index_final = np.random.choice(len(weights), p=weights)
                                    # selected_index_final = tem_selected_index_list[selected_index_final]

                                    # 下面这一部分是聚成两类，在多的那一类中选一个
                                    tem_output_text_list = [output_text_list[ddi] for ddi in tem_selected_index_list] # tem_selected_index_list中元素是在keep_foresight_list中的下标
                                    tem_advantages_list = [advantages_list[ddi] for ddi in tem_selected_index_list]
                                    vectorizer =TfidfVectorizer()
                                    X=vectorizer.fit_transform(tem_output_text_list)
                                    k = 2 # TODO:后面可能会改
                                    kmeans = KMeans(n_clusters=k)
                                    kmeans.fit(X)
                                    cluster_labels = kmeans.labels_
                                    cluster_list = [[] for _ in range(k)]

                                    for aidx, cluster_label in enumerate(cluster_labels):
                                        cluster_list[cluster_label].append(aidx)
                                    cluster_list = [sorted(cluster) for cluster in cluster_list]
                                    cluster0 = []
                                    cluster1 = []
                                    for aidx, cluster_label in enumerate(cluster_labels):
                                        if cluster_label == 0:
                                            cluster0.append(aidx)
                                        else:
                                            cluster1.append(aidx)
                                    if len(cluster0) > len(cluster1):
                                        cluster_adv_list = [tem_advantages_list[ddi] for ddi in cluster0]
                                        weights = softmax([adv/temp for adv in cluster_adv_list])
                                        selected_index_in_cluster = np.random.choice(len(weights), p=weights)
                                        selected_index_in_tem = cluster0[selected_index_in_cluster]
                                        selected_index_final = tem_selected_index_list[selected_index_in_tem]
                                    else:
                                        cluster_adv_list = [tem_advantages_list[ddi] for ddi in cluster1]
                                        weights = softmax([adv/temp for adv in cluster_adv_list])
                                        selected_index_in_cluster = np.random.choice(len(weights), p=weights)
                                        selected_index_in_tem = cluster0[selected_index_in_cluster]
                                        selected_index_final = tem_selected_index_list[selected_index_in_tem]
                                    cur_cluster_info['state'] = 'success'
                                    cur_cluster_info['cluster_result_cluster_idx'] = cluster_list
                                    cur_cluster_info['cluster_result_answer_idx'] = [[tem_selected_index_list[iodx] for iodx in cluster] for cluster in cluster_list]
                                except:
                                    # 利用adv no replace
                                    cur_cluster_info['state'] = 'fail'
                                    weights = softmax([adv/temp for adv in advantages_list])
                                    selected_index_final = np.random.choice(len(weights), p=weights)
                                cur_cluster_info['selected_idx_in_origin'] = selected_index_final
                            traj_info['final_part']['cluster'] = cur_cluster_info
                            # traj_info['selected_idx_in_origin'] = [keep_foresight_list[iiiidx] for iiiidx in selected_index_list]

                        if args.strategy == "sir":
                            temp = 0.1
                            def softmax(x):
                                e_x = np.exp(np.array(x))
                                return e_x / e_x.sum(axis=0)
                            weights = softmax([logp/temp for logp in normalized_logp_list])
                            selected_index_final = np.random.choice(len(weights), p=weights)
                        if args.strategy == "sir_no_replace":
                            temp = 0.1
                            def softmax(x):
                                e_x = np.exp(np.array(x))
                                return e_x / e_x.sum(axis=0)
                            weights = softmax([logp/temp for logp in normalized_logp_list])
                            selected_index_final = np.random.choice(len(weights), p=weights, replace=False)
                        elif args.strategy == "adv":
                            temp = 0.1
                            def softmax(x):
                                e_x = np.exp(np.array(x))
                                return e_x / e_x.sum(axis=0)
                            weights = softmax([logp/temp for logp in advantages_list])
                            selected_index_final = np.random.choice(len(weights), p=weights)
                        elif args.strategy == "adv_no_replace":
                            temp = 0.1
                            def softmax(x):
                                e_x = np.exp(np.array(x))
                                return e_x / e_x.sum(axis=0)
                            weights = softmax([logp/temp for logp in advantages_list])
                            selected_index_final = np.random.choice(len(weights), p=weights, replace=False)
                    else:
                        if args.final_select_strategy == "max_logprob":
                            max_idx = 0
                            
                            max_logprob_pool = []
                            for idx, each_step_prob_list in enumerate(each_step_prob_pool):
                                # 找到step_beam_size个最大的logprob
                                tem_max = -INF
                                for each_step_logprob in each_step_prob_list:
                                    if each_step_logprob > tem_max:
                                        tem_max = each_step_logprob
                                max_logprob_pool.append(tem_max)
                            
                            # 从max_logprob_pool中找到最大的
                            max_logprob = -INF
                            for idx, each_max_logprob in enumerate(max_logprob_pool):
                                if each_max_logprob > max_logprob:
                                    max_logprob = each_max_logprob
                                    max_idx = idx
                            
                            selected_index_final = max_idx

                        
                        elif args.final_select_strategy == "min_logprob":
                            min_idx = 0
                            
                            min_logprob_pool = []
                            for idx, each_step_prob_list in enumerate(each_step_prob_pool):
                                # 找到step_beam_size个最小的logprob
                                tem_min = INF
                                for each_step_logprob in each_step_prob_list:
                                    if each_step_logprob < tem_min:
                                        tem_min = each_step_logprob
                                min_logprob_pool.append(tem_min)
                            
                            # 从max_logprob_pool中找到最大的
                            min_logprob = -INF
                            for idx, each_min_logprob in enumerate(min_logprob_pool):
                                if each_min_logprob > min_logprob:
                                    min_logprob = each_min_logprob
                                    min_idx = idx
                            
                            selected_index_final = min_idx

                        elif args.final_select_strategy == "max_min_diff":
                            max_min_diff_idx = 0
                            
                            max_min_diff_pool = []
                            for idx, each_step_prob_list in enumerate(each_step_prob_pool):
                                # 找到step_beam_size个最大的logprob
                                tem_max = -INF
                                for each_step_logprob in each_step_prob_list:
                                    if each_step_logprob > tem_max:
                                        tem_max = each_step_logprob
                                
                                # 找到step_beam_size个最小的logprob
                                tem_min = INF
                                for each_step_logprob in each_step_prob_list:
                                    if each_step_logprob < tem_min:
                                        tem_min = each_step_logprob
                                
                                max_min_diff_pool.append(tem_max - tem_min)
                            
                            # 从max_logprob_pool中找到最大的
                            max_min_diff = -INF
                            for idx, each_max_min_diff in enumerate(max_min_diff_pool):
                                if each_max_min_diff > max_min_diff:
                                    max_min_diff = each_max_min_diff
                                    max_min_diff_idx = idx
                            
                            selected_index_final = max_min_diff_idx

                        elif args.final_select_strategy == "step_avg":
                            # 把每一步的avg再avg
                            avg_idx = 0
                            
                            avg_logprob_pool = []
                            for idx, each_step_prob_list in enumerate(each_step_prob_pool):
                                tem_avg = sum(each_step_prob_list) / len(each_step_prob_list)
                                avg_logprob_pool.append(tem_avg)
                            
                            avg_logprob = -INF
                            for idx, each_avg_logprob in enumerate(avg_logprob_pool):
                                if each_avg_logprob > avg_logprob:
                                    avg_logprob = each_avg_logprob
                                    avg_idx = idx
                            
                            selected_index_final = avg_idx

                        elif args.final_select_strategy == "sum_avg":
                            # 把所有的求和之后再avg
                            sum_avg_idx = 0

                            sum_avg_logprob_pool = []
                            for idx, each_cumulative_prob_list in enumerate(all_cumulative_logprob_list):
                                tem_sum = sum(each_cumulative_prob_list)
                                token_num = sum(token_num_list[idx])
                                sum_avg_logprob_pool.append(tem_sum/(token_num+1e-8))

                            sum_avg_logprob = -INF
                            for idx, each_sum_avg_logprob in enumerate(sum_avg_logprob_pool):
                                if each_sum_avg_logprob > sum_avg_logprob:
                                    sum_avg_logprob = each_sum_avg_logprob
                                    sum_avg_idx = idx

                            selected_index_final = sum_avg_idx
                        elif args.final_select_strategy == "ar":
                            selected_index_final = 0
                    break

            if args.final_select_strategy == "ar":
                selected_index_final = 0
            # sample_id_pool.append([selected_index_final for _ in range(step_beam_size)])
            whole_traj = previous_steps_list[selected_index_final] + "\n" + candidates_list[selected_index_final] # 最终答案的traj
            whole_traj_list = [previous_steps_list[beam_idx] + "\n" + candidates_list[beam_idx] for beam_idx in range(step_beam_size)] # 每个beam的traj
            traj_info['token_num'] = all_token_num
            all_traj_info.append(traj_info)
                # --- 记录结束时间 ---

            ###########################################
            #           Write to result file          #
            ###########################################
            result = {}
            result['id'] = i
            result['question'] = test_data[i]['question']
            result['ground_truth'] = test_data[i]['answer']
            result['passage'] = test_data[i]['passage']
            result['label'] = test_data[i].get('label', None)
            result['source_ds'] = test_data[i].get('source_ds', None)
            result['response'] = whole_traj
            result['response_all_beams'] = whole_traj_list
            all_input_token_num += input_token_num_for_this_question
            all_output_token_num += output_token_num_for_this_question
            
            if args.record_process:

                with open(args.time_path+"TRAJ_INFO-"+ffname+'.json', 'w') as fa:
                    fa.write(json.dumps(all_traj_info) + '\n')
            f.write(json.dumps(result) + '\n')
            f.flush()

            end_time = time.perf_counter()
            end_dt = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S.%f")[:-3]
            elapsed_ms = (end_time - start_time) * 1000
            print(f"[INFO] Finished processing at {end_dt}")
            print(f"[INFO] Total processing time: {elapsed_ms:.2f} ms")

            print('input_token_num_for_this_question: ', input_token_num_for_this_question)
            print('output_token_num_for_this_question: ', output_token_num_for_this_question)
            print('all_input_token_num: ', all_input_token_num)
            print('all_output_token_num: ', all_output_token_num)
    end_time = time.time()
    time_span = end_time - start_time
    print(f"time: {time_span}")
    # ffname = 'sigmarate_' + args.sigma_rate + '-' +  args.pruning_strategy
    time_path = args.time_path + ffname +  '.txt'
    with open(time_path, 'a') as f:
        f.write('time:  ' + str(time_span) + '\n')
        f.write('total:  ' + str(total_rollout_times) + '\n')
        f.write('save:  ' + str(saved_rollout_times) + '\n')
        f.write('num_rollout:  ' + str(num_rollout) + '\n')
        f.write('num_foresight:  ' + str(num_foresight) + '\n')
        f.write('step_beam_size:  ' + str(step_beam_size) + '\n')
        f.write('strategy:  ' + str(args.strategy) + '\n')
        f.write('final_select_strategy:  ' + str(args.final_select_strategy) + '\n')
        f.write('pruning_strategy:  ' + str(args.pruning_strategy) + '\n')
        f.write('depth_pruning_strategy:  ' + str(args.depth_pruning_strategy) + '\n')
        f.write('threshold:  ' + str(args.threshold) + '\n')
        f.write('sigma_rate:  ' + str(args.sigma_rate) + '\n')
        f.write('adv_above0:  ' + str(args.adv_above0) + '\n')
        f.write('all_input_token_num:  ' + str(all_input_token_num) + '\n')
        f.write('all_output_token_num:  ' + str(all_output_token_num) + '\n')
    print('total: ', total_rollout_times)
    print('save: ', saved_rollout_times)