import os
import json
import argparse
from tqdm import tqdm
import logging
import pdb
import os
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))
from benchmarks.strong_reject.strong_reject.evaluate import strongreject_finetuned, strongreject_finetuned_small_batch

def load_data(file_path):
    """Load JSON data from the specified file path."""
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Data file not found: {file_path}")
    with open(file_path, 'r', encoding='utf-8') as f:
        return json.load(f)


def extract_prompts_and_responses(data):
    """Extract prompts and model responses from the dataset."""
    prompts, responses = [], []
    for item in tqdm(data, desc="Extracting Prompts and Responses"):
        prompts.append(item["vanilla prompt"])
        responses.append(item['response'])
    return prompts, responses


def classify_jailbreaks(prompts, responses, evaluator_path):
    """Classify using StrongREJECT fine-tuned model."""
    batch = {"forbidden_prompt": prompts, "response": responses}
    # result = strongreject_finetuned(batch)
    result = strongreject_finetuned_small_batch(batch, evaluator_path)
    scores = result["score"]
    average_score = sum(scores) / len(scores)
    # classifications = [1 if score > 0.5 else 0 for score in scores]
    return average_score, scores


def main(args):
    logging.basicConfig(level=logging.INFO, format='[%(asctime)s] %(levelname)s: %(message)s')
    # Load data
    logging.info("Loading data...")
    data = load_data(args.data_path)

    # Extract prompts and responses
    logging.info("Extracting prompts and responses...")
    prompts, responses = extract_prompts_and_responses(data)

    # Classify jailbreak responses
    logging.info("Classifying jailbreak responses using StrongREJECT...")
    average_score, scores = classify_jailbreaks(prompts, responses, args.evaluator_path)

    # Output results
    # print("Scores:", scores)
    logging.info(f"Average score: {average_score}")

    # save results
    os.makedirs(os.path.dirname(args.output_path), exist_ok=True)
    for i, item in enumerate(tqdm(data, total=len(data), desc="Saving results")):
        item['asr_e_label'] = scores[i]
    with open(args.output_path, 'w', encoding='utf-8') as f:
        json.dump(data, f, indent=4, ensure_ascii=False)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Classify jailbreak behavior in model responses")
    parser.add_argument(
        "--data-path", "-d",
        type=str,
        required=True,
        help="Path to the JSON file containing generated responses"
    )

    parser.add_argument(
        "--evaluator-path", "-e",
        type=str,
        required=True,
        help="Path to the StrongREJECT fine-tuned model evaluator"
    )

    parser.add_argument(
        "--output-path", "-o",
        type=str,
        required=True,
        help="Path to save the output JSON file with jailbreak classifications"
    )

    args = parser.parse_args()
    main(args)