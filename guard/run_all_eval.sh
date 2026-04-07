#!/bin/bash
# Run TokenGuard on all 7 datasets (26 examples each), then evaluate
set -e

GUARD_DIR="/data/rhq/rhq945/Token-Guard/guard"
DATA_DIR="/data/rhq/rhq945/Token-Guard/data"
EVAL_DIR="/data/rhq/rhq945/Token-Guard/eval"
OUT_DIR="/data/rhq/rhq945/Token-Guard/results/tg_run_$(date +%Y%m%d_%H%M%S)"
MODEL="/data/rhq/TOKEN-GUARD/models/Meta-Llama-3.1-8B-Instruct"

mkdir -p "$OUT_DIR/raw" "$OUT_DIR/processed"

cd "$GUARD_DIR"

declare -A DATASET_MAP
DATASET_MAP["CovidQA"]="covidQA"
DATASET_MAP["DROP_History"]="history"
DATASET_MAP["DROP_Nfl"]="nfl"
DATASET_MAP["FinanceBench"]="financebench"
DATASET_MAP["Halueval"]="halueval"
DATASET_MAP["PubmedQA"]="pubmedqa"
DATASET_MAP["RAGTruth"]="ragtruth"

for JSON_FILE in "$DATA_DIR"/*.json; do
    BASENAME=$(basename "$JSON_FILE" .json)
    DS_NAME="${DATASET_MAP[$BASENAME]:-${BASENAME,,}}"
    OUT_FILE="$OUT_DIR/raw/${BASENAME}.json"

    echo ""
    echo "=========================================="
    echo "Running: $BASENAME  (dataset=$DS_NAME)"
    echo "=========================================="

    CUDA_VISIBLE_DEVICES=0 conda run -n halu python run_guard.py \
        --model_path "$MODEL" \
        --datasets "$DS_NAME" \
        --data_path "$JSON_FILE" \
        --max_examples 26 \
        --num_rollout 1 \
        --num_foresight 2 \
        --step_beam_size 1 \
        --shot_mode fewshot \
        --tau_global 0.65 \
        --output_dir "$OUT_DIR/raw/" \
        --time_path "$OUT_DIR/time/" \
        --file_name "$BASENAME" \
        2>&1 | tee "$OUT_DIR/raw/${BASENAME}_log.txt"

    echo "Done: $BASENAME"
done

echo ""
echo "=========================================="
echo "Extracting answers (process.py)"
echo "=========================================="
python3 "$EVAL_DIR/processed_answer/process.py" \
    --input_dir "$OUT_DIR/raw" \
    --output_dir "$OUT_DIR/processed" 2>/dev/null || \
python3 - << PYEOF
import os, json, re

def extract_answer(text, dataset_name=''):
    # For CovidQA: use last "Answer:" occurrence (handles multi-Answer: responses)
    if 'CovidQA' in dataset_name or 'covidqa' in dataset_name.lower():
        return extract_answer_covidqa(text)
    m = re.search(r'(?:Answer|answer)\s*[:：]\s*"?([^"\]\}\n]+)"?', text)
    if m:
        ans = m.group(1).strip().rstrip('.')
        return ans
    return text.strip()[:200] if text.strip().lower() != 'cannot answer' else ''

def extract_answer_covidqa(resp):
    """CovidQA-specific extraction: use LAST Answer: occurrence, allow multi-line."""
    # Find last "Answer:" occurrence
    last_pos = -1
    for marker in ['Answer:', 'answer:', 'Answer：', 'answer：']:
        pos = resp.rfind(marker)
        if pos > last_pos:
            last_pos = pos
    if last_pos >= 0:
        after = resp[last_pos:]
        # Strip the marker
        after = re.sub(r'^[Aa]nswer\s*[:：]\s*', '', after).strip()
        # Strip surrounding quotes
        after = after.strip('"\'')
        # Take up to 3 sentences or 500 chars, whichever is shorter
        sentences = re.split(r'(?<=[.!?])\s+', after)
        candidate = ' '.join(sentences[:3]).strip()
        # Cap at 500 chars
        return candidate[:500]
    # Fallback: last non-empty line that isn't a reasoning step header
    lines = [l.strip() for l in resp.split('\n') if l.strip() and not l.strip().startswith('The reasoning')]
    if lines:
        return lines[-1][:400]
    return resp.strip()[:400]

in_dir  = "$OUT_DIR/raw"
out_dir = "$OUT_DIR/processed"
os.makedirs(out_dir, exist_ok=True)

for fname in os.listdir(in_dir):
    if not fname.endswith('.json') or '_log' in fname:
        continue
    results = []
    with open(os.path.join(in_dir, fname)) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                item = json.loads(line)
                pred = extract_answer(item.get('response', ''), dataset_name=fname)
                results.append({
                    'id':           item.get('id', ''),
                    'passage':      item.get('passage', ''),
                    'question':     item.get('question', ''),
                    'ground_truth': item.get('ground_truth', ''),
                    'answer':       pred
                })
            except:
                pass
    out_path = os.path.join(out_dir, fname.replace('.json', '_processed.json'))
    with open(out_path, 'w') as f:
        for r in results:
            f.write(json.dumps(r, ensure_ascii=False) + '\n')
    print(f'Processed {fname}: {len(results)} items')
PYEOF

echo ""
echo "=========================================="
echo "Evaluating (eval.py)"
echo "=========================================="
python3 - << PYEOF
import os, json, re, numpy as np
from scipy.optimize import linear_sum_assignment
import string

def _norm(text):
    text = text.lower()
    text = re.sub(r'\b(a|an|the)\b', ' ', text)
    text = ''.join(c for c in text if c not in string.punctuation or c in '.%')
    return ' '.join(text.split())

def compute_f1(pred, gold):
    p_toks = set(_norm(pred).split())
    g_toks = set(_norm(gold).split())
    if not p_toks or not g_toks:
        return 0.0
    inter = len(p_toks & g_toks)
    prec  = inter / len(p_toks)
    rec   = inter / len(g_toks)
    return 2*prec*rec/(prec+rec)*100 if (prec+rec) > 0 else 0.0

def compute_em(pred, gold):
    return 1.0 if _norm(pred) == _norm(gold) else 0.0

processed_dir = "$OUT_DIR/processed"
results_summary = {}

for fname in sorted(os.listdir(processed_dir)):
    if not fname.endswith('.json'):
        continue
    dataset = fname.replace('_processed.json', '')
    em_list, f1_list, cannot_list = [], [], []
    with open(os.path.join(processed_dir, fname)) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                item = json.loads(line)
                pred  = item.get('answer', '')
                gold  = item.get('ground_truth', '')
                cannot_list.append(1 if pred == '' else 0)
                em_list.append(compute_em(pred, gold))
                f1_list.append(compute_f1(pred, gold))
            except:
                pass

    if em_list:
        em  = np.mean(em_list)
        f1  = np.mean(f1_list)
        cannot_rate = np.mean(cannot_list)
        results_summary[dataset] = {'EM': em, 'F1': f1, 'cannot_answer_rate': cannot_rate, 'n': len(em_list)}
        print(f'{dataset:20s}  EM={em:.4f}  F1={f1:.2f}  cannot_answer={cannot_rate:.1%}  n={len(em_list)}')

# Print avg
if results_summary:
    avg_em = np.mean([v['EM'] for v in results_summary.values()])
    avg_f1 = np.mean([v['F1'] for v in results_summary.values()])
    print(f'\n{"AVG":20s}  EM={avg_em:.4f}  F1={avg_f1:.2f}')

# Save to file
with open('$OUT_DIR/eval_summary.json', 'w') as f:
    json.dump(results_summary, f, indent=2)
print(f'\nSaved to $OUT_DIR/eval_summary.json')
PYEOF

echo ""
echo "All done. Results in: $OUT_DIR"
