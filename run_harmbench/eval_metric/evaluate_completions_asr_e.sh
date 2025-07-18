#!/bin/bash

# source /opt/rh/devtoolset-10/enable

source ~/miniconda3/etc/profile.d/conda.sh
conda activate harmbench

cd ~/diffusion_lm/llada_safety/HarmBench

model_name=$1
attack_method=$2
defense_method=$3

cls_path="/mnt/petrelfs/wenzichen/hf_models/HarmBench-Llama-2-13b-cls"
behaviors_path="/mnt/petrelfs/wenzichen/diffusion_lm/llada_safety/HarmBench/data/behavior_datasets/harmbench_behaviors_text_all_refined_v3.csv"
completions_path="/mnt/petrelfs/wenzichen/diffusion_lm/llada_safety/HarmBench/data/generated_completions_${model_name}_${attack_method}_attack_${defense_method}_v3.json"
save_path="./results/eval_results_${model_name}_${attack_method}_attack_${defense_method}_v3.json"


python -u evaluate_completions.py \
    --cls_path $cls_path \
    --behaviors_path $behaviors_path \
    --completions_path $completions_path \
    --save_path $save_path \
    --include_advbench_metric

echo "cls_path=$cls_path"
echo "behaviors_path=$behaviors_path"
echo "completions_path=$completions_path"
echo "save_path=$save_path"
