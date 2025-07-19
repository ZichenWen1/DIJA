import os
import re
import sys
import json
import torch
import argparse
import logging
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModel

import pandas as pd
import torch.nn.functional as F
import torch.distributions as dists
from typing import Optional, Union, Dict, Any, Tuple, List
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "../"))
sys.path.insert(0, PROJECT_ROOT)
from defense_utils import Defender

# Constants
DEFAULT_GEN_LENGTH = 64
DEFAULT_STEPS = 64
DEFAULT_MASK_ID = 151666
DEFAULT_MASK_COUNTS = 36
MASK_TOKEN = "<|mask|>"
SPECIAL_TOKEN_PATTERN = r'<mask:(\d+)>'
COLOR_BLUE = "\033[94m"
COLOR_RESET = "\033[0m"


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description="Generate responses using LLaDA model with optional defense")
    parser.add_argument("--model_path", type=str, required=True)
    parser.add_argument("--attack_prompt", type=str, required=True)
    parser.add_argument("--output_json", type=str, required=True)
    parser.add_argument("--steps", type=int, default=DEFAULT_STEPS)
    parser.add_argument("--gen_length", type=int, default=DEFAULT_GEN_LENGTH)
    parser.add_argument("--mask_id", type=int, default=DEFAULT_MASK_ID)
    parser.add_argument("--mask_counts", type=int, default=DEFAULT_MASK_COUNTS)
    parser.add_argument("--attack_method", type=str, default="zeroshot")
    parser.add_argument("--defense_method", type=str, default=None)
    return parser.parse_args()


def process_prompt_instruct(prompt: str, mask_counts: int) -> Tuple[str, int]:
    """Replace <mask:x> with repeated <|mask|> and pad if necessary."""
    def replace(match):
        return "<|mask|>" * int(match.group(1))

    prompt = re.sub(SPECIAL_TOKEN_PATTERN, replace, prompt)

    if "<|mask|>" not in prompt and mask_counts:
        prompt += "<|mask|>" * mask_counts

    return prompt, prompt.count("<|mask|>")


def prepare_prompt(tokenizer, behavior: str, is_instruct: bool) -> str:
    """Prepare prompt based on whether model is Instruct-style."""
    if is_instruct:
        return tokenizer.apply_chat_template(
            [{"role": "user", "content": behavior}],
            tokenize=False,
            add_generation_prompt=True
        )
    else:
        return behavior


def generate_response(
    vanilla_prompt: str,
    prompt: str,
    tokenizer,
    model,
    args
) -> str:
    """Generate a response given a prompt using diffusion-style generation."""
    input_ids = tokenizer(prompt)["input_ids"]
    attention_mask = tokenizer(prompt)["attention_mask"]

    vanilla_input_ids = tokenizer(vanilla_prompt)["input_ids"]
    vanilla_prompt_len = len(vanilla_input_ids)

    matching_count = next(
        (i for i, (a, b) in enumerate(zip(vanilla_input_ids, input_ids)) if a != b),
        min(len(vanilla_input_ids), len(input_ids))
    )

    input_ids = torch.tensor(input_ids).unsqueeze(0).to(model.device)
    attention_mask = torch.tensor(attention_mask).unsqueeze(0).to(model.device).float()

    output_ids = model.diffusion_generate(
        input_ids,
        attention_mask=attention_mask,
        max_new_tokens=args.gen_length,
        output_history=False,
        return_dict_in_generate=True,
        steps=args.steps,
        temperature=0.2,
        top_p=0.95,
    ).sequences

    response = tokenizer.batch_decode(output_ids[:, matching_count:], skip_special_tokens=True)[0]

    if args.attack_method == "DIJA":
        response = response.split("assistant\n")[0]

    return response


def main():
    logging.basicConfig(level=logging.INFO, format='[%(asctime)s] %(levelname)s: %(message)s')
    args = parse_args()
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    # Initialize defender if needed
    defender = Defender(args.defense_method) if args.defense_method and args.defense_method.lower() != "none" else None

    # Determine instruct model
    is_instruct_model = "Instruct" in args.model_path or "1.5" in args.model_path

    # Load model and tokenizer
    tokenizer = AutoTokenizer.from_pretrained(args.model_path, trust_remote_code=True)
    model = AutoModel.from_pretrained(
        args.model_path,
        trust_remote_code=True,
        torch_dtype=torch.float32
    ).to(device).eval()

    # Load attack prompts
    with open(args.attack_prompt, 'r') as f:
        data = json.load(f)

    results = []

    with torch.no_grad():
        for idx, item in enumerate(tqdm(data, desc="Processing data", total=len(data))):
            behavior_id = item['BehaviorID']
            vanilla_behavior = item['Behavior']
            refined_behavior = item.get('Refined_behavior', '')

            # Choose behavior
            if args.attack_method == "DIJA" or "refine" in args.attack_prompt:
                behavior = refined_behavior
            else:
                behavior = vanilla_behavior

            # Prepare prompts
            prompt = prepare_prompt(tokenizer, behavior, is_instruct_model)
            vanilla_prompt = prepare_prompt(tokenizer, vanilla_behavior, is_instruct_model)

            # Process masking
            if args.attack_method == "DIJA":
                prompt, _ = process_prompt_instruct(prompt, args.mask_counts)

            # Apply defense if specified
            if defender:
                prompt = defender.handler(prompt)
                vanilla_prompt = defender.handler(vanilla_prompt)
                response = generate_response(vanilla_prompt, prompt, tokenizer, model, args)
            else:
                response = generate_response(vanilla_prompt, prompt, tokenizer, model, args)

            logging.info(f"{COLOR_BLUE}Response: {response}{COLOR_RESET}\n")

            # Save result
            results.append({
                "BehaviorID": behavior_id,
                "Behavior": item['Behavior'],
                "Refined_behavior": refined_behavior,
                "response": response
            })

    # Save JSON
    os.makedirs(os.path.dirname(args.output_json), exist_ok=True)
    with open(args.output_json, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=4, ensure_ascii=False)
    logging.info(f"Saved JSON to {args.output_json}")


if __name__ == "__main__":
    main()
