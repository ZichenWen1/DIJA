#!/bin/bash



cd $HOME/DIJA/run_jailbreakbench


model_name=$1
attack_method=$2
defense_method=$3
version=$4

JSON_PATH=""


echo "Running ASR-k computation..."
echo "Input JSON path: $JSON_PATH"


python evaluate_completions_asr_k.py \
    --json_path "$JSON_PATH" \

