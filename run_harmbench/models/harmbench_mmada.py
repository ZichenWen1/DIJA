import os
import re
import sys
import json
import torch
import argparse
import logging
from tqdm import tqdm
from transformers import AutoTokenizer

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../../")))
from MMaDA.models import MMadaModelLM


sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../")))
from utility.generate_function import generate_mmada
from defense_utils import Defender


DEFAULT_GEN_LENGTH = 128
DEFAULT_STEPS = 128
DEFAULT_MASK_ID = 126336
DEFAULT_MASK_COUNTS = 36
MASK_TOKEN = "<|mdm_mask|>"
START_TOKEN = "<startoftext>"
END_TOKEN = "<endoftext>"
SPECIAL_TOKEN_PATTERN = r'<mask:(\d+)>'
COLOR_BLUE = "\033[94m"
COLOR_RESET = "\033[0m"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate responses using MMaDA model")
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


def process_prompt_instruct(prompt: str, mask_counts: int) -> str:
    def replace_mask(match):
        return MASK_TOKEN * int(match.group(1))

    prompt = re.sub(SPECIAL_TOKEN_PATTERN, replace_mask, prompt)

    if MASK_TOKEN not in prompt and mask_counts:
        prompt += START_TOKEN + MASK_TOKEN * mask_counts + END_TOKEN

    return prompt


def get_tokenized_input(prompt: str, tokenizer, device: str):
    inputs = tokenizer(prompt, return_tensors="pt", padding=True, truncation=True)
    return inputs["input_ids"].to(device), inputs["attention_mask"].to(device)


def generate_response(
    vanilla_prompt: str,
    prompt: str,
    tokenizer,
    model,
    args
) -> str:
    input_ids, attention_mask = get_tokenized_input(prompt, tokenizer, model.device)
    vanilla_input_ids, _ = get_tokenized_input(vanilla_prompt, tokenizer, model.device)

    matching_count = min(
        sum(a == b for a, b in zip(vanilla_input_ids[0], input_ids[0])),
        len(vanilla_input_ids[0])
    )

    output_ids = generate_mmada(
        input_ids=input_ids,
        attention_mask=attention_mask,
        model=model,
        steps=args.steps,
        gen_length=args.gen_length,
        block_length=args.steps,
        temperature=0.2,
        mask_id=args.mask_id
    )

    response = tokenizer.batch_decode(output_ids[:, matching_count:], skip_special_tokens=True)[0]
    if args.attack_method == "DIJA":
        response = response.split("assistant\n")[0]

    return response


def prepare_prompt(tokenizer, behavior: str, is_mmada: bool) -> str:
    if is_mmada:
        return tokenizer.apply_chat_template(
            [{"role": "user", "content": behavior}],
            tokenize=False,
            add_generation_prompt=True
        )
    else:
        return behavior


def main():
    logging.basicConfig(level=logging.INFO, format='[%(asctime)s] %(levelname)s: %(message)s')
    args = parse_args()
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    # Initialize defender if needed
    defender = Defender(args.defense_method) if args.defense_method and args.defense_method.lower() != "none" else None
    is_mmada_model = "MixCoT" in args.model_path

    tokenizer = AutoTokenizer.from_pretrained(args.model_path, trust_remote_code=True)
    model = MMadaModelLM.from_pretrained(
        args.model_path,
        trust_remote_code=True,
        torch_dtype=torch.bfloat16
    ).to(device).eval()

    # Chat template for MMaDA
    # tokenizer.chat_template = (
    #     "{% set loop_messages = messages %}"
    #     "{% for message in loop_messages %}"
    #     "{% set content = '<|start_header_id|>' + message['role'] + '<|end_header_id|>\n' + message['content'] | trim + '<|eot_id|>' %}"
    #     "{% if loop.index0 == 0 %}{% set content = bos_token + content %}{% endif %}"
    #     "{{ content }}"
    #     "{% endfor %}"
    #     "{{ '<|start_header_id|>assistant<|end_header_id|>\n' }}"
    # )

    with open(args.attack_prompt, 'r') as f:
        data = json.load(f)

    results = []

    with torch.no_grad():
        for idx, item in enumerate(tqdm(data, desc="Processing data", total=len(data))):
            behavior_id = item['BehaviorID']
            vanilla_behavior = item['Behavior']
            refined_behavior = item.get('Refined_behavior', '')

            behavior = (
                refined_behavior if args.attack_method == "DIJA" or "refine" in args.attack_prompt
                else vanilla_behavior
            )

            prompt = prepare_prompt(tokenizer, behavior, is_mmada_model)
            vanilla_prompt = prepare_prompt(tokenizer, vanilla_behavior, is_mmada_model)
            prompt = process_prompt_instruct(prompt, args.mask_counts)

            if defender:
                prompt = defender.handler(prompt)
                vanilla_prompt = defender.handler(vanilla_prompt)

            response = generate_response(vanilla_prompt, prompt, tokenizer, model, args)

            logging.info(f"{COLOR_BLUE}[{idx}] Response: {response}{COLOR_RESET}\n")

            results.append({
                "BehaviorID": behavior_id,
                "Behavior": vanilla_behavior,
                "Refined_behavior": refined_behavior,
                "response": response
            })

    os.makedirs(os.path.dirname(args.output_json), exist_ok=True)
    with open(args.output_json, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=4, ensure_ascii=False)
    logging.info(f"Saved JSON to {args.output_json}")


if __name__ == "__main__":
    main()
