#!/bin/bash


cd $HOME/DIJA/run_jailbreakbench

model_name=$1 # victim model--dLLMs
attack_method=$2
defense_method=$3
version=$4

INPUT_FILE="$HOME/DIJA/run_jailbreakbench/eval_results/eval_results_${model_name}_${attack_method}_attack_${defense_method}_defense_${version}.json"
OUTPUT_FILE="$HOME/DIJA/run_jailbreakbench/eval_results/harmfulness_score/${model_name}_${attack_method}_${defense_method}_${version}.json"
JUDGE_MODEL="gpt-4o"
POLICY_MODEL="gpt-4o"
NUM_PROCESSES=100
API_KEY=""  # TODO: Set your OpenAI API key here
BASE_URL="" # TODO: Set your OpenAI API base URL here if needed


while getopts ":i:o:j:p:m:d:k:h" opt; do
  case $opt in
    i) INPUT_FILE="$OPTARG" ;;
    j) JUDGE_MODEL="$OPTARG" ;;
    p) POLICY_MODEL="$OPTARG" ;;
    m) NUM_PROCESSES="$OPTARG" ;;
    d) DEBUG_MODE="$OPTARG" ;;
    k) API_KEY="$OPTARG" ;;
    h)
      echo "Usage: $0 [options]"
      echo "Options:"
      echo "  -i <input_file>       Input JSON file (default: data.json)"
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


python3 harmfulscore.py \
  --input_file "${INPUT_FILE}" \
  --output_file "${OUTPUT_FILE}" \
  --judge_model "${JUDGE_MODEL}" \
  --policy_model "${POLICY_MODEL}" \
  --num_processes "${NUM_PROCESSES}" \
  --api_key "${API_KEY}" \
  --base_url "${BASE_URL}" \

echo "Evaluation completed. Results saved to ${OUTPUT_FILE}"