import os
import json
import argparse
from tqdm import tqdm
import sys
import os

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))
from benchmarks.jailbreakbench.src.jailbreakbench.classifier import Llama3JailbreakJudge


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
        prompts.append(item["goal"])
        responses.append(item['response'])
    return prompts, responses


def classify_jailbreaks(prompts, responses, api_key):
    classifier = Llama3JailbreakJudge(api_key)
    return classifier.classify_responses(prompts, responses)


def main(args):
    # Load data
    print("Loading data...")
    data = load_data(args.data_path)

    # Extract prompts and responses
    print("Extracting prompts and responses...")
    prompts, responses = extract_prompts_and_responses(data)

    # Classify jailbreak responses
    print("Classifying jailbreak responses...")
    classifications = classify_jailbreaks(prompts, responses, args.api_key)

    # Output results
    num_jailbroken = sum(classifications)
    total = len(prompts)
    print(f"{num_jailbroken}/{total} Jailbroken (Total of {total} submitted prompts.)")

    # save results
    os.makedirs(os.path.dirname(args.output_path), exist_ok=True)
    for i, item in enumerate(tqdm(data, total=len(data), desc="Saving results")):
        item['asr_e_label'] = classifications[i]
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
        "--api-key", "-a",
        type=str,
        required=True
    )

    parser.add_argument(
        "--output-path", "-o",
        type=str,
        required=True,
        help="Path to save the output JSON file with jailbreak classifications"
    )

    args = parser.parse_args()
    main(args)