import os
import json
import re
import glob
from datetime import datetime


def extract_simple_answer(line: str):
    pattern = r'(?:answer|Answer)\s*[:：]\s*"?([^"\]\}\n]+)"?'
    matches = re.findall(pattern, line)
    if not matches:
        return ""
    content = matches[-1].strip()
    if content.endswith("."):
        content = content[:-1].strip()
    return (content) 


def process_result_file(file_path, output_dir, log_file=None):
    """处理单个结果文件"""
    filename = os.path.basename(file_path)
    print(f"处理文件: {filename}")
    if log_file:
        log_file.write(f"处理文件: {filename}\n")
    
    processed_results = []
    success_count = 0
    error_count = 0
    
    with open(file_path, 'r', encoding='utf-8') as f:
        lines = f.read().splitlines()
        
        print(f"  总共 {len(lines)} 行，开始处理...")
        if log_file:
            log_file.write(f"  总共 {len(lines)} 行，开始处理...\n")
        
        for i, line in enumerate(lines):
            line = line.strip()
            if not line:
                continue
            
            try:
                item = json.loads(line)
            except json.JSONDecodeError:
                error_count += 1
                if error_count <= 5:
                    print(f"    第{i+1}行: JSON解析失败")
                    if log_file:
                        log_file.write(f"    第{i+1}行: JSON解析失败\n")
                continue
            
            item_id = item.get("id", f"line_{i+1}")
            question = item.get("question", "")
            passage = item.get("passage", "")
            ground_truth = item.get("ground_truth", "")
            
            predicted_answer = extract_simple_answer(line)
            
            processed_results.append({
                "id": item_id,
                "passage": passage,
                "question": question,
                "ground_truth": ground_truth,
                "answer": predicted_answer
            })
            success_count += 1
    
    # 保存结果
    output_filename = filename.replace('.json', '_processed.json')
    output_path = os.path.join(output_dir, output_filename)
    
    with open(output_path, "w", encoding="utf-8") as f:
        for result in processed_results:
            f.write(json.dumps(result, ensure_ascii=False, indent=2) + "\n\n")
    
    print(f"  处理完成: 成功 {success_count} 个，失败 {error_count} 个")
    print(f"  保存到: {output_path}")
    if log_file:
        log_file.write(f"  处理完成: 成功 {success_count} 个，失败 {error_count} 个\n")
        log_file.write(f"  保存到: {output_path}\n")
    return processed_results


def main():
    # 自动获取最新的时间戳文件夹作为输入目录
    base_results_dir = "/data/rhq/TOKEN-GUARD/halu/results"
    if not os.path.exists(base_results_dir):
        print(f"基础目录不存在: {base_results_dir}")
        return
    
    # 查找所有baselines_开头的文件夹
    baseline_dirs = []
    for item in os.listdir(base_results_dir):
        if item.startswith("baselines_") and os.path.isdir(os.path.join(base_results_dir, item)):
            baseline_dirs.append(item)
    
    if not baseline_dirs:
        print(f"在 {base_results_dir} 中没有找到baselines_开头的文件夹")
        return
    
    # 按时间戳排序，获取最新的
    baseline_dirs.sort()
    latest_baseline = baseline_dirs[-1]
    results_dir = os.path.join(base_results_dir, latest_baseline, "results")
    
    if not os.path.exists(results_dir):
        print(f"结果目录不存在: {results_dir}")
        return
    
    # 使用当前日期作为输出目录
    current_date = datetime.now().strftime("%m%d")
    output_dir = f"/data/rhq/TOKEN-GUARD/halu/eval/processed_answer/{current_date}"
    
    # 创建输出目录
    os.makedirs(output_dir, exist_ok=True)
    
    # 创建日志文件
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file_path = os.path.join(output_dir, f"processing_log_{timestamp}.txt")
    
    # 查找所有结果文件
    result_files = glob.glob(os.path.join(results_dir, "*.json"))
    
    if not result_files:
        print(f"在 {results_dir} 中没有找到JSON文件")
        return
    
    print(f"使用输入目录: {results_dir}")
    print(f"使用输出目录: {output_dir}")
    print(f"找到 {len(result_files)} 个结果文件")
    print(f"日志文件: {log_file_path}")
    
    # 处理每个文件
    with open(log_file_path, 'w', encoding='utf-8') as log_file:
        log_file.write(f"处理开始时间: {datetime.now()}\n")
        log_file.write(f"输入目录: {results_dir}\n")
        log_file.write(f"输出目录: {output_dir}\n")
        log_file.write(f"找到 {len(result_files)} 个结果文件\n")
        
        for file_path in result_files:
            try:
                process_result_file(file_path, output_dir, log_file)
            except Exception as e:
                error_msg = f"处理文件 {file_path} 时出错: {e}"
                print(error_msg)
                log_file.write(error_msg + "\n")
        
        log_file.write(f"处理结束时间: {datetime.now()}\n")
    
    print("所有文件处理完成！")
    print(f"详细日志已保存到: {log_file_path}")


if __name__ == "__main__":
    main()
