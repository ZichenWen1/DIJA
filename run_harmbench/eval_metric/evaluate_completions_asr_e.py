import transformers
from transformers import AutoTokenizer, AutoModelForCausalLM
import json
import argparse
import os
import numpy as np
from tqdm import tqdm
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))
from benchmarks.HarmBench.eval_utils import (
    compute_results_hashing,
    compute_results_classifier,
    compute_results_advbench,
    compute_results_classifier_without_vllm,
)

transformers.utils.logging.set_verbosity_error()

def parse_args():
    parser = argparse.ArgumentParser(description="Running red teaming evaluation.")
    parser.add_argument("--cls_path", type=str, default='cais/HarmBench-Llama-2-13b-cls')
    parser.add_argument("--completions_path", type=str, required=True)
    parser.add_argument("--save_path", type=str, required=True)
    parser.add_argument("--include_advbench_metric", action="store_true")
    parser.add_argument("--num_tokens", type=int, default=512)
    return parser.parse_args()

def main():
    args = parse_args()
    print(args)

    # ========== Load completions (JSON list) ========== #
    with open(args.completions_path, 'r') as f:
        completions = json.load(f)

    # ========== Load classifier model and tokenizer ========== #
    cls_model = AutoModelForCausalLM.from_pretrained(args.cls_path, device_map="cuda")
    cls_tokenizer = AutoTokenizer.from_pretrained(args.cls_path)
    cls_tokenizer.truncation_side = "left"

    results = {}

    success_rates = []
    adv_success_rates = []

    for item in tqdm(completions, desc="Scoring completions", total=len(completions)):
        behavior_id = item["BehaviorID"]
        behavior_text = item["Behavior"]
        tags = []  

        # Clip response length
        response = item["response"]
        tokenized = cls_tokenizer.encode(response, max_length=args.num_tokens, truncation=True)
        clipped_response = cls_tokenizer.decode(tokenized, skip_special_tokens=True)
        item["generation"] = clipped_response  

        behavior_dict = {
            "BehaviorID": behavior_id,
            "Behavior": behavior_text,
            "Tags": ", ".join(tags) if tags else "", 
        }

        if 'hash_check' in tags:
            current_results = compute_results_hashing(behavior_dict, [item])
        else:
            current_results = compute_results_classifier_without_vllm(
                behavior_dict, [item], cls_model, cls_tokenizer
            )

        if args.include_advbench_metric:
            current_results = compute_results_advbench(current_results, [item])

        results[behavior_id] = current_results

        labels = [d.get("asr_e_label", 0) for d in current_results]
        adv_labels = [d.get("advbench_label", 0) for d in current_results if "advbench_label" in d]

        success_rates.append(np.mean(labels))
        if adv_labels:
            adv_success_rates.append(np.mean(adv_labels))

    print(f"Average ASR: {np.mean(success_rates):.4f}")
    if adv_success_rates:
        print(f"Average AdvBench ASR: {np.mean(adv_success_rates):.4f}")

    # === save results ===
    os.makedirs(os.path.dirname(args.save_path), exist_ok=True)
    with open(args.save_path, 'w') as f:
        json.dump(results, f, indent=4)
    print(f"Saved evaluation results to {args.save_path}")

if __name__ == "__main__":
    main()
