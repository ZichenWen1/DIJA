#!/bin/bash


source ~/miniconda3/etc/profile.d/conda.sh
conda activate harmbench

cd ~/diffusion_lm/LLaDA/run_harmbench

model_name=$1
attack_method=$2
defense_method=$3

version=$4

INPUT_FILE="/mnt/petrelfs/wenzichen/diffusion_lm/llada_safety/HarmBench/data/generated_completions_${model_name}_${attack_method}_attack_${defense_method}_${version}.json"
OUTPUT_DIR="results"
OUTPUT_FILE="/mnt/petrelfs/wenzichen/diffusion_lm/LLaDA/run_harmbench/HS_results/${model_name}_${attack_method}_${defense_method}_${version}.json"
JUDGE_MODEL="gpt-4o"
POLICY_MODEL="gpt-4o"
NUM_PROCESSES=400
API_KEY=sk-dPUeaXawcoYL7ZkQ8bd89QHgUvXmYF1SboQs7n80uKiqTKcZ


while getopts ":i:o:j:p:m:d:k:h" opt; do
  case $opt in
    i) INPUT_FILE="$OPTARG" ;;
    o) OUTPUT_DIR="$OPTARG" ;;
    j) JUDGE_MODEL="$OPTARG" ;;
    p) POLICY_MODEL="$OPTARG" ;;
    m) NUM_PROCESSES="$OPTARG" ;;
    d) DEBUG_MODE="$OPTARG" ;;
    k) API_KEY="$OPTARG" ;;
    h)
      echo "Usage: $0 [options]"
      echo "Options:"
      echo "  -i <input_file>       Input JSON file (default: data.json)"
      echo "  -o <output_dir>       Output directory (default: results)"
      echo "  -j <judge_model>      Judge model name (default: gpt-4o)"
      echo "  -p <policy_model>     Policy model name (default: gpt-4o)"
      echo "  -m <num_processes>    Number of processes (default: 1)"
      echo "  -d <debug>            Debug mode (true/false, default: false)"
      echo "  -k <api_key>          API key for GPT inference (default: EMPTY)"
      echo "  -h                    Show help"
      exit 0
      ;;
    \?)
      echo "Invalid option: -$OPTARG" >&2
      exit 1
      ;;
    :)
      echo "Option -$OPTARG requires an argument." >&2
      exit 1
      ;;
  esac
done


mkdir -p "${OUTPUT_DIR}"


TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
if [ -z "${OUTPUT_FILE}" ]; then
  OUTPUT_FILE="${OUTPUT_DIR}/results_${TIMESTAMP}.json"
fi


python3 harmfulscore.py \
  --input_file "${INPUT_FILE}" \
  --output_file "${OUTPUT_FILE}" \
  --judge_model "${JUDGE_MODEL}" \
  --policy_model "${POLICY_MODEL}" \
  --num_processes "${NUM_PROCESSES}" \
  --api_key "${API_KEY}"


echo "Evaluation completed. Results saved to ${OUTPUT_FILE}"