#!/bin/bash

# --- 配置 ---
ts=$(date +%Y%m%d_%H%M%S)
output_dir="results/baselines_$ts"
log_dir="$output_dir/log"
result_dir="$output_dir/results"
time_dir="$output_dir/time"
memory_log_dir="$output_dir/memory_logs"

single_python="/data/rhq/TOKEN-GUARD/halu/baselines/final/guard/zeroshot-phi_decoding.py"

# --- 创建目录 ---
mkdir -p "$log_dir" "$result_dir" "$time_dir" "$memory_log_dir"

echo "Result directory: $result_dir"
echo "Time directory: $time_dir"
echo "Memory log directory: $memory_log_dir"

# --- 主循环（完全串行） ---
# for baseline_py in baselines/final/guard/*.py; do
for baseline_py in "$single_python"; do
  if [[ -f "$baseline_py" ]]; then
    tag=$(basename "$baseline_py" .py)
    echo "Processing baseline: $baseline_py ..."

    for f in data2/*.json; do
      if [[ -f "$f" ]]; then
        fname=$(basename "$f" .json)
        echo "  Processing data file: $f ..."

        # --- 定义文件路径 ---
        log_file="$log_dir/${tag}_${fname}_$ts.log"
        memory_log_file="$memory_log_dir/${tag}_${fname}_$ts.mem.log"

        echo "  Output log: $log_file"
        echo "  Memory log: $memory_log_file"

        {
          echo "[$(date)] Starting job: ${tag}_${fname}"

          # 1) 启动 python（前台运行）
          CUDA_VISIBLE_DEVICES=0 python -u "$baseline_py" \
            --data_path "$f" \
            --output_dir "$result_dir" \
            --file_name "${tag}_${fname}" \
            --model_id "llama3.1" \
            --time_path "$time_dir" \
            --max_examples 30 &

          python_pid=$!
          echo "Python PID: $python_pid"

          # 2) 启动 pidstat（后台）
          if command -v pidstat >/dev/null 2>&1; then
            echo "[$(date)] Memory monitor PIDSTAT started."
            pidstat -r -p "$python_pid" 1 >> "$memory_log_file" 2>&1 &
            pidstat_pid=$!
          else
            echo "PIDSTAT not found" >> "$memory_log_file"
          fi

          # 3) 等待 python 完成
          wait "$python_pid"
          python_exit=$?
          echo "[$(date)] Python process exited with code $python_exit"

          # 4) 停止 pidstat
          if [[ -n "$pidstat_pid" ]]; then
            kill "$pidstat_pid" 2>/dev/null || true
            wait "$pidstat_pid" 2>/dev/null || true
          fi

          echo "[$(date)] Finished job: ${tag}_${fname}"
        } > "$log_file" 2>&1

      fi
    done
  fi
done

echo "全部处理完成：$output_dir"
