#!/bin/bash

source ~/miniconda3/etc/profile.d/conda.sh
conda activate harmbench

cd ~/diffusion_lm/LLaDA/run_harmbench

# DEFAULT_JSON_PATH="/mnt/petrelfs/wenzichen/diffusion_lm/llada_safety/HarmBench/data/generated_completions_llada_1.5_ours_attack.json"

# JSON_PATH=${1:-$DEFAULT_JSON_PATH}


model_name=$1
attack_method=$2
defense_method=$3
late_threshold_ratio=$4

version=$5

JSON_PATH="/mnt/petrelfs/wenzichen/diffusion_lm/llada_safety/HarmBench/data/generated_completions_${model_name}_${attack_method}_attack_${defense_method}_${version}.json"


echo "Running keyword-based ASR computation..."
echo "Input JSON path: $JSON_PATH"


python evaluate_completions_asr_k.py \
    --json_path "$JSON_PATH" \


