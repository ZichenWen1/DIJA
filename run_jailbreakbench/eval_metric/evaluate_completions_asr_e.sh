#!/bin/bash

cd $HOME/DIJA/run_jailbreakbench

jailbreak_model_name=$1
attack_mathod=$2
defense_method=$3
version=$4

DATA_PATH="/mnt/petrelfs/wenzichen/diffusion_lm/llada_safety/jailbreakbench/generated_completions_${jailbreak_model_name}_${attack_mathod}_attack_${defense_method}_${version}.json"
MODEL_PATH="/mnt/petrelfs/wenzichen/hf_models/HarmBench-Llama-2-13b-cls" # similar to harmbench

python eval_response.py \
    --data-path "${DATA_PATH}" \
    --model-path "${MODEL_PATH}"

echo "${DATA_PATH}"
echo "${MODEL_PATH}"
