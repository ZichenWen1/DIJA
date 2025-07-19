#!/usr/bin/env python3
"""
Refine HarmBench prompts using either a local HF model or a remote OpenAI-style API.
Usage:
    python refiner.py --help
    python refiner.py hf --hf-model-path Qwen/Qwen2.5-7B-Instruct --attack-prompt prompts.csv --output-json out.json
"""
from __future__ import annotations

import argparse
import random
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any, Dict, List, Optional

import pandas as pd
import torch
from openai import OpenAI
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer


def load_prompt_template(path: str) -> str:
    with open(path, 'r', encoding='utf-8') as f:
        return f.read()


def apply_prompt_template(vanilla_prompt: str, template: str) -> str:
    return template.format(prompt=vanilla_prompt)


class Refiner(ABC):
    """
    Base class for refining behaviors with LLMs.
    Subclasses must implement `_refine_single(behavior: str) -> str`.
    """

    def __init__(
        self,
        prompt_template_path: str,
        attack_prompt: str,
        output_json: str,
        max_new_tokens: int = 100,
    ):
        self.template = load_prompt_template(prompt_template_path)
        self.df = pd.read_csv(attack_prompt)
        self.out_file = Path(output_json)
        self.max_new_tokens = max_new_tokens

    @abstractmethod
    def _refine_single(self, behavior: str) -> str:
        """Return a refined version of the input behavior."""
        ...

    def run(self) -> None:
        refined: List[str] = []
        try:
            for _, row in tqdm(
                self.df.iterrows(), total=len(self.df), desc="Refining behaviors"
            ):
                refined.append(self._refine_single(row["Behavior"]))
        except KeyboardInterrupt:
            print("\nInterrupted by user, saving partial results…")
        self.df["Refined_behavior"] = refined
        self.out_file.parent.mkdir(parents=True, exist_ok=True)
        self.df.to_json(self.out_file, orient="records", indent=2, force_ascii=False)
        print(f"✅ Saved refined behaviors to {self.out_file}")

    # ------------------------------------------------------------------
    # CLI wiring
    # ------------------------------------------------------------------
    @classmethod
    def build_arg_parser(cls) -> argparse.ArgumentParser:
        p = argparse.ArgumentParser(description="Refine HarmBench prompts using LLM")

        # Add backend subparsers
        sub = p.add_subparsers(dest="backend", required=True)

        # Common arguments for both backends
        def add_common_args(parser):
            parser.add_argument("--prompt-template-path", required=True)
            parser.add_argument("--attack-prompt", required=True, help="CSV with 'Behavior' column")
            parser.add_argument("--output-json", required=True)
            parser.add_argument("--max-new-tokens", type=int, default=100)

        # HF backend
        hf = sub.add_parser("hf", help="Use local Hugging Face model")
        hf.add_argument("--hf-model-path", required=True)
        add_common_args(hf)

        # API backend
        api = sub.add_parser("api", help="Use remote API model")
        api.add_argument("--api-model-name", required=True)
        api.add_argument("--base-url", required=True)
        api.add_argument("--api-key")
        api.add_argument("--hf-fallback-path", required=True)
        add_common_args(api)

        return p

    @classmethod
    def dispatch(cls) -> None:
        parser = cls.build_arg_parser()
        args = parser.parse_args()

        kwargs: Dict[str, Any] = {
            "prompt_template_path": args.prompt_template_path,
            "attack_prompt": args.attack_prompt,
            "output_json": args.output_json,
            "max_new_tokens": args.max_new_tokens,
        }

        if args.backend == "hf":
            HFRefiner(args.hf_model_path, **kwargs).run()
        elif args.backend == "api":
            APIRefiner(
                args.api_model_name,
                args.base_url,
                args.api_key,
                args.hf_fallback_path,
                **kwargs,
            ).run()
        else:
            parser.error("Unknown backend")


class HFRefiner(Refiner):
    def __init__(
        self,
        hf_model_path: str,
        prompt_template_path: str,
        attack_prompt: str,
        output_json: str,
        max_new_tokens: int = 100,
    ):
        super().__init__(
            prompt_template_path=prompt_template_path,
            attack_prompt=attack_prompt,
            output_json=output_json,
            max_new_tokens=max_new_tokens,
        )
        self.hf_model_path = hf_model_path
        self.model = AutoModelForCausalLM.from_pretrained(
            hf_model_path, torch_dtype="auto", device_map="auto"
        )
        self.tokenizer = AutoTokenizer.from_pretrained(hf_model_path)

    def _refine_single(self, behavior: str) -> str:
        prompt = apply_prompt_template(behavior, self.template)
        return self._chat_generate(prompt)

    @torch.inference_mode()
    def _chat_generate(self, prompt: str) -> str:
        messages = [
            {
                "role": "system",
                "content": "You are a helpful assistant and strictly follow the instructions.",
            },
            {"role": "user", "content": prompt},
        ]
        text = self.tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        inputs = self.tokenizer([text], return_tensors="pt").to(self.model.device)
        gen = self.model.generate(
            **inputs, max_new_tokens=self.max_new_tokens, do_sample=False
        )
        gen_ids = gen[0][len(inputs.input_ids[0]) :]
        return self.tokenizer.decode(gen_ids, skip_special_tokens=True).strip()


class APIRefiner(Refiner):
    def __init__(
        self,
        api_model_name: str,
        base_url: str,
        api_key: Optional[str],
        hf_fallback_path: str,
        prompt_template_path: str,
        attack_prompt: str,
        output_json: str,
        max_new_tokens: int = 100,
    ):
        super().__init__(
            prompt_template_path=prompt_template_path,
            attack_prompt=attack_prompt,
            output_json=output_json,
            max_new_tokens=max_new_tokens,
        )
        self.api_model_name = api_model_name
        self.client = OpenAI(base_url=base_url, api_key=api_key)
        self.fallback = HFRefiner(hf_fallback_path, **{
            "prompt_template_path": prompt_template_path,
            "attack_prompt": attack_prompt,
            "output_json": output_json,
            "max_new_tokens": max_new_tokens,
        })

    def _refine_single(self, behavior: str) -> str:
        prompt = apply_prompt_template(behavior, self.template)
        return self._try_api_then_fallback(prompt)

    def _try_api_then_fallback(self, prompt: str) -> str:
        for attempt in range(5):
            temp = 0.3 if attempt == 0 else random.uniform(0.2, 0.7)
            try:
                response = (
                    self.client.chat.completions.create(
                        model=self.api_model_name,
                        messages=[{"role": "user", "content": prompt}],
                        temperature=temp,
                        max_tokens=self.max_new_tokens,
                    )
                    .choices[0]
                    .message.content
                )
                if response and "sorry" not in response.lower():
                    return response
            except Exception as e:
                print(f"[API] Attempt {attempt + 1} failed: {e}")
        # All retries exhausted -> use HF fallback
        print("\033[91m[Warning] Using HF fallback\033[0m")
        return self.fallback._chat_generate(prompt)


if __name__ == "__main__":
    Refiner.dispatch()