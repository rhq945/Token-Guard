#!/bin/bash

# --- 1. 环境配置 (采用相对路径) ---
# 确保脚本在项目根目录下执行
ts=$(date +%Y%m%d_%H%M%S)
# 将结果统一存放在 eval/eval_results 下的子目录，这样符合你之前的目录结构
output_dir="eval/eval_results/baselines_$ts"
log_dir="$output_dir/log"
result_dir="$output_dir/results"
time_dir="$output_dir/time"
memory_log_dir="$output_dir/memory_logs"

# 修正后的 Python 脚本相对路径 (根据你之前的 ls 结果)
single_python="guard/run_guard.py"

# --- 2. 创建目录 ---
mkdir -p "$log_dir" "$result_dir" "$time_dir" "$memory_log_dir"

echo "-----------------------------------------------"
echo "Token-Guard Baseline Evaluation Pipeline"
echo "Result Directory: $result_dir"
echo "-----------------------------------------------"

# --- 3. 主循环 ---
# 遍历当前目录下的 data/ 文件夹
for baseline_py in "$single_python"; do
  if [[ -f "$baseline_py" ]]; then
    tag=$(basename "$baseline_py" .py)
    echo "🚀 Processing baseline: $tag ..."

    # 修正：将 data2 改为 data
    for f in data/*.json; do
      if [[ -f "$f" ]]; then
        fname=$(basename "$f" .json)
        echo "  📂 Data file: $fname"

        # 定义日志路径
        log_file="$log_dir/${tag}_${fname}_$ts.log"
        memory_log_file="$memory_log_dir/${tag}_${fname}_$ts.mem.log"

        {
          echo "[$(date)] Starting job: ${tag}_${fname}"

          # 执行推理测试
          # 注意：这里增加了对 --model_path 的支持，如果你的代码需要加载模型
          CUDA_VISIBLE_DEVICES=3 python -u "$baseline_py" \
            --data_path "$f" \
            --output_dir "$result_dir" \
            --file_name "${tag}_${fname}" \
            --model_id "llama3.1" \
            --time_path "$time_dir" \
            --max_examples 30 &

          python_pid=$!
          echo "Python PID: $python_pid"

          # 内存监控 (pidstat)
          if command -v pidstat >/dev/null 2>&1; then
            pidstat -r -p "$python_pid" 1 >> "$memory_log_file" 2>&1 &
            pidstat_pid=$!
          fi

          # 等待当前任务完成（串行执行，保护 A40 显存）
          wait "$python_pid"
          python_exit=$?
          
          if [[ -n "$pidstat_pid" ]]; then
            kill "$pidstat_pid" 2>/dev/null || true
          fi

          echo "[$(date)] Finished job with exit code $python_exit"
        } > "$log_file" 2>&1
      fi
    done
  fi
done

echo "-----------------------------------------------"
echo "🎉 所有实验处理完成：$output_dir"
