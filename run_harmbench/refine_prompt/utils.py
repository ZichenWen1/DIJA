import pandas as pd
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from tqdm import tqdm
import random
from openai import OpenAI
import logging
import os
import json


def load_prompt_template(path: str) -> str:
    with open(path, 'r', encoding='utf-8') as f:
        return f.read()


def apply_prompt_template(vanilla_prompt: str, template: str) -> str:
    return template.format(prompt=vanilla_prompt)


class Refiner:
    def __init__(self, hf_model_path, api_model_name, prompt_template_path, attack_prompt, output_json, base_url=None, api_key=None):
        self.model_path = hf_model_path
        self.api_model_name = api_model_name
        self.prompt_template_path = prompt_template_path
        self.attack_prompt = attack_prompt
        self.output_json = output_json
        self.base_url = base_url
        self.api_key = api_key
        self.template_str = self.load_prompt_template(prompt_template_path)

        if self.base_url and self.api_key:
            self.client = OpenAI(base_url=base_url, api_key=api_key)

        self.model = AutoModelForCausalLM.from_pretrained(hf_model_path, torch_dtype="auto", device_map="auto")
        self.tokenizer = AutoTokenizer.from_pretrained(hf_model_path)

    def load_prompt_template(self, path):
        with open(path, 'r', encoding='utf-8') as file:
            return file.read()

    def apply_prompt_template(self, vanilla_prompt, template):
        return template.format(prompt=vanilla_prompt)

    def api_generate(self, prompt, api_model_name, temperature, max_new_tokens=128):
        try:
            response = self.client.chat.completions.create(
                model=api_model_name,
                messages=[
                    {
                        "role": "user",
                        "content": [
                            {"type": "text", "text": prompt},
                        ]
                    }
                ],
                temperature=temperature,
                max_tokens=max_new_tokens
            )
            return response.choices[0].message.content
        except Exception as e:
            logging.error(f"API call failed: {e}")
            return None

    def qwen_generate(self, prompt, max_new_tokens):
        messages = [
            {"role": "system", "content": "You are a helpful assistant and strictly follow the instructions."},
            {"role": "user", "content": prompt}
        ]
        text = self.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        model_inputs = self.tokenizer([text], return_tensors="pt").to(self.model.device)

        generated_ids = self.model.generate(
            **model_inputs,
            max_new_tokens=max_new_tokens
        )
        generated_ids = [output_ids[len(input_ids):] for input_ids, output_ids in zip(model_inputs.input_ids, generated_ids)]
        response = self.tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]
        return response.strip()

    def run_refinement_hf(self, max_new_tokens=100):
        df = pd.read_csv(self.attack_prompt)
        refined_list = []

        with torch.no_grad():
            for idx, row in tqdm(df.iterrows(), total=len(df), desc="Refining Behaviors"):
                vanilla_behavior = row['Behavior']
                prompt = self.apply_prompt_template(vanilla_behavior, self.template_str)
                response = self.qwen_generate(prompt, max_new_tokens)
                logging.info(f"[Refined]: {response}")
                refined_list.append(
                    {
                        "BehaviorID": row['BehaviorID'],
                        "FunctionalCategory": row['FunctionalCategory'],
                        "SemanticCategory": row['SemanticCategory'],
                        "Behavior": row['Behavior'],
                        "Refined_behavior": response,
                    }
                )

        os.makedirs(os.path.dirname(self.output_json), exist_ok=True)
        with open(self.output_json, 'w', encoding='utf-8') as f:
            json.dump(refined_list, f, ensure_ascii=False, indent=4)
        logging.info(f"Saved refined behaviors to {self.output_json}")

    def run_refinement_api(self, max_new_tokens=120):
        df = pd.read_csv(self.attack_prompt)
        refined_list = []

        for idx, row in tqdm(df.iterrows(), total=len(df), desc="Refining Behaviors"):
            vanilla_behavior = row['Behavior']
            prompt = self.apply_prompt_template(vanilla_behavior, self.template_str)

            for i in range(5):
                if i == 0:
                    response = self.api_generate(prompt, self.api_model_name, temperature=0.3, max_new_tokens=max_new_tokens)
                else:
                    temperature_sample = random.uniform(0.2, 0.7)
                    response = self.api_generate(prompt, self.api_model_name, temperature=temperature_sample, max_new_tokens=max_new_tokens)
                if (response is not None) and ("sorry" not in response.lower()):
                    break

            if (response is None) or ("sorry" in response.lower()):
                logging.warning("Using local Qwen model")
                response = self.qwen_generate(prompt, max_new_tokens)

            logging.info(f"[Refined]: {response}")

            refined_list.append(
                {
                    "BehaviorID": row['BehaviorID'],
                    "FunctionalCategory": row['FunctionalCategory'],
                    "SemanticCategory": row['SemanticCategory'],
                    "Behavior": row['Behavior'],
                    "Refined_behavior": response,
                }
            )

        os.makedirs(os.path.dirname(self.output_json), exist_ok=True)
        with open(self.output_json, 'w', encoding='utf-8') as f:
            json.dump(refined_list, f, ensure_ascii=False, indent=4)
        logging.info(f"Saved refined behaviors to {self.output_json}")
