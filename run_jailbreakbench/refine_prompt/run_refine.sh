#!/bin/bash

# export CUDA_VISIBLE_DEVICES=0,1,2,3

cd $HOME/DIJA/run_jailbreakbench/refine_prompt

version=$1
hf_model_path="$HOME/DIJA/hf_models/Qwen2.5-7B-Instruct"

# TODO: Set your API key and base URL if using an API model
api_model_name=""
api_key="" # TODO: set your API key here
base_url="" # TODO: set your API base URL here

prompt_template_path="$HOME/DIJA/run_jailbreakbench/refine_prompt/redteam_prompt_template.txt"
attack_prompt=""
output_json="$HOME/DIJA/run_jailbreakbench/refine_prompt/jailbreakbench_data_refined_${version}.json"
max_new_tokens=200


python main.py \
  --hf_model_path "${hf_model_path}" \
  --api_model_name "${api_model_name}" \
  --prompt_template_path "${prompt_template_path}"  \
  --attack_prompt "${attack_prompt}"  \
  --output_json "${output_json}" \
  --base_url "${base_url}" \
  --api_key "${API_KEY}" \
  --max_new_tokens "${max_new_tokens}"