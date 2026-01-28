import os
import json
import re
import time
import string
import numpy as np
from scipy.optimize import linear_sum_assignment
import nltk
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
from rouge_score import rouge_scorer

smooth_fn = SmoothingFunction().method1
rouge_l_scorer = rouge_scorer.RougeScorer(['rougeL'], use_stemmer=True)

# -------------------- 文本处理 & 归一化 -------------------- #
def _remove_articles(text):
    return re.sub(r"\b(a|an|the)\b", " ", text)

def _white_space_fix(text):
    return " ".join(text.split())

def _is_number(text):
    try:
        float(text)
        return True
    except ValueError:
        pass
    if text.endswith('%'):
        try:
            float(text[:-1])
            return True
        except ValueError:
            pass
    return False

def _remove_punc(text):
    if not _is_number(text):
        return "".join(ch for ch in text if ch not in set(string.punctuation))
    else:
        return text

def _lower(text):
    return text.lower()

def _tokenize(text):
    return re.split(" |-", text)

def _normalize_number(text):
    if _is_number(text):
        if text.endswith('%'):
            return str(float(text[:-1]))
        else:
            return str(float(text))
    else:
        return text

def _normalize_answer(text):
    parts = [
        _white_space_fix(
            _remove_articles(
                _normalize_number(_remove_punc(_lower(token)))
            )
        )
        for token in _tokenize(text)
    ]
    parts = [part for part in parts if part.strip()]
    return " ".join(parts).strip()

# -------------------- F1 / EM / BLEU / ROUGE-L 计算 -------------------- #
def _answer_to_bags(answer):
    if isinstance(answer, (list, tuple)):
        raw_spans = answer
    else:
        raw_spans = [answer]
    normalized_spans = []
    token_bags = []
    for raw_span in raw_spans:
        normalized_span = _normalize_answer(raw_span)
        normalized_spans.append(normalized_span)
        token_bags.append(set(normalized_span.split()))
    return normalized_spans, token_bags

def _compute_f1(predicted_bag, gold_bag):
    intersection = len(gold_bag.intersection(predicted_bag))
    precision = intersection / float(len(predicted_bag)) if predicted_bag else 1.0
    recall = intersection / float(len(gold_bag)) if gold_bag else 1.0
    f1 = (2 * precision * recall) / (precision + recall) if not (precision == 0.0 and recall == 0.0) else 0.0
    return f1 * 100

def _match_numbers_if_present(gold_bag, predicted_bag):
    gold_numbers = set(word for word in gold_bag if _is_number(word))
    predicted_numbers = set(word for word in predicted_bag if _is_number(word))
    if (not gold_numbers) or gold_numbers.intersection(predicted_numbers):
        return True
    return False

def _align_bags(predicted, gold):
    scores = np.zeros([len(gold), len(predicted)])
    for gold_index, gold_item in enumerate(gold):
        for pred_index, pred_item in enumerate(predicted):
            if _match_numbers_if_present(gold_item, pred_item):
                scores[gold_index, pred_index] = _compute_f1(pred_item, gold_item)
    row_ind, col_ind = linear_sum_assignment(-scores)
    max_scores = np.zeros([max(len(gold), len(predicted))])
    for row, column in zip(row_ind, col_ind):
        max_scores[row] = max(max_scores[row], scores[row, column])
    return max_scores

def compute_bleu(pred, gold):
    if not isinstance(gold, list):
        gold = [gold]
    references = [_normalize_answer(g).split() for g in gold]
    candidate = _normalize_answer(pred).split()
    score = sentence_bleu(references, candidate, smoothing_function=smooth_fn, weights=(1,0,0,0))  # unigram BLEU
    return score * 100

def compute_rouge_l(pred, gold):
    score = rouge_l_scorer.score(gold, pred)
    return score['rougeL'].fmeasure * 100

def get_metrics(predicted, gold):
    predicted_bags = _answer_to_bags(predicted)
    gold_bags = _answer_to_bags(gold)
    
    exact_match = 1.0 if set(predicted_bags[0]) == set(gold_bags[0]) and len(predicted_bags[0]) == len(gold_bags[0]) else 0.0
    f1_per_bag = _align_bags(predicted_bags[1], gold_bags[1])
    f1 = round(np.mean(f1_per_bag), 4)
    
    bleu_scores = []
    rouge_l_scores = []
    for p, g in zip(predicted_bags[0], gold_bags[0]):
        bleu_scores.append(compute_bleu(p, g))
        rouge_l_scores.append(compute_rouge_l(p, g))
    bleu = round(np.mean(bleu_scores), 4)
    rouge_l = round(np.mean(rouge_l_scores), 4)
    
    return exact_match, f1, bleu, rouge_l

# -------------------- 主评估函数 -------------------- #
def eval_all_json_in_dir(result_dir):
    ts = time.strftime('%Y%m%d_%H%M%S')
    output_file = os.path.join(result_dir, f'drop_eval_result_{ts}.txt')
    lines = []

    for filename in sorted(os.listdir(result_dir)):
        # 跳过非 JSON 或包含 "error" 的文件
        if not filename.endswith('.json') or 'error' in filename.lower():
            continue
        
        file_path = os.path.join(result_dir, filename)
        em_total, f1_total, bleu_total, rouge_total, count = 0, 0, 0, 0, 0

        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()
            # 正则匹配每个 JSON 对象
            objs = re.findall(r'\{[^}]*\}', content, re.DOTALL)
            for obj_str in objs:
                try:
                    item = json.loads(obj_str)
                except Exception as e:
                    print(f"解析失败: {e}\n内容片段: {obj_str[:100]}")
                    continue

                pred = item.get('answer', '')
                gold = item.get('ground_truth', '')

                em, f1, bleu, rouge_l = get_metrics(pred, gold)

                em_total += em
                f1_total += f1
                bleu_total += bleu
                rouge_total += rouge_l
                count += 1

        if count > 0:
            result_line = f"{filename}: EM={em_total/count:.4f}, F1={f1_total/count:.4f}, BLEU={bleu_total/count:.4f}, ROUGE-L={rouge_total/count:.4f}, Total={count}"
        else:
            result_line = f"{filename}: No valid data."
        print(result_line)
        lines.append(result_line)

    # 写入最终结果
    with open(output_file, 'w', encoding='utf-8') as fout:
        for l in lines:
            fout.write(l + '\n')

    print(f"评估结果已写入: {output_file}")

# -------------------- 脚本入口 -------------------- #
if __name__ == "__main__":
    eval_all_json_in_dir('/data/rhq/TOKEN-GUARD/halu/eval/processed_answer/1206')
