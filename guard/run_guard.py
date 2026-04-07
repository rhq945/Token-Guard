"""
run_guard.py — Entry point for TokenGuard decoding.
Parses arguments and drives the per-example processing loop.

Module layout:
  generation_utils.py  — softmax, TEMPERATURE, TokenGuardGenerator
  prompt_builder.py    — PromptBuilder (passage preprocessing + prompts)
  beam_search.py       — BeamSearchEngine (clustering, step processing, early stopping)
  decoder.py           — TokenGuardDecoder (model loading, process_example)
  token_guard_plugin.py — LatentEnvironment, TokenGuardConfig, CandidateSegment (unchanged)
  logic_example.py     — few-shot prompt constants (unchanged)
"""
import argparse
import json
import os
import time

from decoder import TokenGuardDecoder


def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="TokenGuard Decoding Algorithm")

    # Model configuration
    parser.add_argument('--model_id', type=str, default='llama3.1')
    parser.add_argument(
        '--model_path',
        type=str,
        default='/data/rhq/TOKEN-GUARD/halu/models/Meta-Llama-3.1-8B-Instruct',
    )
    parser.add_argument('--gpus', type=int, default=1)

    # Data configuration
    parser.add_argument('--datasets', type=str, default='gsm')
    parser.add_argument(
        '--data_path',
        type=str,
        default='/data/rhq/TOKEN-GUARD/halu/data2/halueval.json',
    )
    parser.add_argument('--output_dir', type=str, default='./results/')

    # Algorithm parameters
    parser.add_argument('--step_beam_size', type=int, default=4)
    parser.add_argument('--num_rollout', type=int, default=10)
    parser.add_argument('--num_foresight', type=int, default=8)
    parser.add_argument('--strategy', type=str, default='cluster')
    parser.add_argument('--width_pruning_strategy', type=str, default='low_sigma')
    parser.add_argument('--depth_pruning_strategy', type=str, default='cluster')
    parser.add_argument('--cluster_num', type=int, default=2)
    parser.add_argument('--threshold', type=float, default=0.75)
    parser.add_argument('--least_foresight_num', type=int, default=4)
    parser.add_argument('--sigma_rate', type=float, default=0.8)

    # Execution configuration
    parser.add_argument('--record_process', type=bool, default=True)
    parser.add_argument('--file_name', type=str, default='test_3')
    parser.add_argument('--time_path', type=str, default='./results/time/')
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--max_examples', type=int, default=50)
    parser.add_argument(
        '--shot_mode', type=str, default='fewshot', choices=['zeroshot', 'fewshot']
    )
    parser.add_argument(
        '--tau_global',
        type=float,
        default=None,
        help='Override global convergence threshold (paper default: 0.7; practical: 0.65)',
    )
    return parser.parse_args()


def main():
    args = parse_arguments()
    decoder = TokenGuardDecoder(args)

    with open(args.data_path) as f:
        test_data = json.load(f)
    max_num = (
        len(test_data) if args.max_examples == -1 else min(len(test_data), args.max_examples)
    )
    os.makedirs(args.output_dir, exist_ok=True)
    os.makedirs(args.time_path, exist_ok=True)
    output_path = os.path.join(args.output_dir, f"{args.file_name}.json")

    start_time = time.time()
    total_stats = {
        "total_rollouts": 0,
        "saved_rollouts": 0,
        "input_tokens": 0,
        "output_tokens": 0,
    }
    all_traj_info = []

    for i, example in enumerate(test_data[:max_num]):
        print(f"[{i+1}/{max_num}] Question: {example.get('question', '')[:50]}")

        system_prompt = decoder.get_system_prompt()
        result = decoder.process_example(example, system_prompt)

        total_stats["total_rollouts"] += result["rollout_stats"]["total"]
        total_stats["saved_rollouts"] += result["rollout_stats"]["saved"]
        total_stats["input_tokens"] += result["token_stats"]["input"]
        total_stats["output_tokens"] += result["token_stats"]["output"]

        result["traj_info"]["question_idx"] = i
        all_traj_info.append(result["traj_info"])

        output_result = {
            "id": i,
            "question": example["question"],
            "passage": example["passage"],
            "ground_truth": example.get("answer"),
            "response": result["response"],
            "response_all_beams": result["trajectories"]
            .get("final", {})
            .get("responses", []),
        }

        with open(output_path, "a") as f:
            f.write(json.dumps(output_result) + "\n")

        print(f'output_token_num_for_question{i}: {result["token_stats"]["output"]}')
        print(f'input_token_num_for_question{i}: {result["token_stats"]["input"]}')
        print(f'all_output_token_num: {total_stats["output_tokens"]}')
        print(f'all_input_token_num: {total_stats["input_tokens"]}')

        if args.record_process:
            traj_path = os.path.join(args.time_path, f"TRAJ_INFO-{args.file_name}.json")
            with open(traj_path, "w") as f:
                json.dump(all_traj_info, f, indent=2)

    time_span = time.time() - start_time
    time_info_path = os.path.join(args.time_path, f"{args.file_name}.txt")
    with open(time_info_path, "w") as f:
        f.write(f'time:  {time_span}\n')
        f.write(f'total:  {total_stats["total_rollouts"]}\n')
        f.write(f'save:  {total_stats["saved_rollouts"]}\n')
        f.write(f'num_rollout:  {args.num_rollout}\n')
        f.write(f'num_foresight:  {args.num_foresight}\n')
        f.write(f'step_beam_size:  {args.step_beam_size}\n')
        f.write(f'strategy:  {args.strategy}\n')
        f.write(f'width_pruning_strategy:  {args.width_pruning_strategy}\n')
        f.write(f'depth_pruning_strategy:  {args.depth_pruning_strategy}\n')
        f.write(f'threshold:  {args.threshold}\n')
        f.write(f'sigma_rate:  {args.sigma_rate}\n')
        f.write(f'cluster_num:  {args.cluster_num}\n')
        f.write(f'all_input_token_num:  {total_stats["input_tokens"]}\n')
        f.write(f'all_output_token_num:  {total_stats["output_tokens"]}\n')

    print('total rollouts: ', total_stats["total_rollouts"])
    print('saved rollouts: ', total_stats["saved_rollouts"])
    print('all_output_token_num: ', total_stats["output_tokens"])
    print('all_input_token_num: ', total_stats["input_tokens"])


if __name__ == "__main__":
    main()
