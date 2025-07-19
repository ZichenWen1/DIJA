#!/bin/bash

source ~/miniconda3/etc/profile.d/conda.sh
conda activate harmbench

cd ~/diffusion_lm/llada_safety/strong_reject/run_strongreject

jailbreak_model_name=$1
attack_mathod=$2
defense_method=$3

DATA_PATH="/mnt/petrelfs/wenzichen/diffusion_lm/llada_safety/strong_reject/generated_completions_${jailbreak_model_name}_${attack_mathod}_attack_${defense_method}.json"


python eval_response.py \
    --data-path "${DATA_PATH}" \

echo "${DATA_PATH}"
echo "${MODEL_PATH}"
