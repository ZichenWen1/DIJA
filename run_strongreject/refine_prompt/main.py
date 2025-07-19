import argparse
import logging
from utils import Refiner  
import os


def main():
    parser = argparse.ArgumentParser(description="Refine HarmBench prompts using LLM")
    parser.add_argument('--hf_model_path', type=str, default=None)
    parser.add_argument('--api_model_name', type=str, default=None)
    parser.add_argument('--prompt_template_path', type=str, default=None)
    parser.add_argument('--attack_prompt', type=str, default=None)
    parser.add_argument('--output_json', type=str, default=None)
    parser.add_argument('--max_new_tokens', type=int, default=100)
    parser.add_argument('--api_key', type=str, default=None)
    parser.add_argument('--base_url', type=str, default=None)

    args = parser.parse_args()

    # Initialize the Refiner object
    refiner = Refiner(
        hf_model_path=args.hf_model_path,
        api_model_name=args.api_model_name,  
        prompt_template_path=args.prompt_template_path,
        attack_prompt=args.attack_prompt,
        output_json=args.output_json,  
        base_url=args.base_url,
        api_key=args.api_key
    )

    # Dynamically choose the refinement method based on the model path
    if args.base_url and args.api_key and args.api_model_name:
        logging.info("Using API model for refinement.")
        refiner.run_refinement_api(max_new_tokens=args.max_new_tokens)
    else:
        logging.info("Using local model for refinement.")
        refiner.run_refinement_hf(max_new_tokens=args.max_new_tokens)

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
    main()