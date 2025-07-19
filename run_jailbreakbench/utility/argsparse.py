import argparse
import os
from pathlib import Path
from datetime import datetime


def parse_args():

    file_path = Path(__file__)
    current_dir = os.path.dirname(file_path)
    home_dir = os.path.dirname(current_dir)


    parser = argparse.ArgumentParser()    
    # Experimental config
    parser.add_argument(
        "--victim_model_name",
        default="gpt-3.5-turbo",
        type=str,
        choices=['o1', 'o1-mini', 'o1-mini-2024-09-12', 'o3-mini', 'gpt-3.5-turbo', 'gpt-4o', 'gpt-4o-mini', 'gpt-4o-2024-05-13', 'gpt-4o-2024-08-06', 'azure-gpt4o', 'gpt-4-1106-preview', 'gpt-4-0613', 'claude-v2', 'llama3-api-8b', 'llama3-api-70b', 'deepseek-r1', 'deepseek-v3'],
        help="The name of chosen victim model. The llama-2 models are called through Deepinfra (see 'llama-api-Xb'); The llama-3 models are called through Siliconflow (see llama3-api-Xb). Note that Deepinfra announced that 'Due to low usage 7b and 13b have been replaced by meta-llama/Meta-Llama-3-8B-Instruct and 70b has been replaced by meta-llama/Meta-Llama-3-70B-Instruct. Your inference requests are still working but they are redirected. Please update your code to use another model.'"
    )
    parser.add_argument(
        "--judge_model_name",
        default="gpt-4-0613",
        type=str,
        choices=['gpt-4o', 'gpt-4o-mini', 'gpt-4o-2024-05-13', 'gpt-4o-2024-08-06', 'azure-gpt4o', 'gpt-4-1106-preview', 'gpt-4-turbo-2024-04-09', 'gpt-4', 'gpt-4-0613'],
        help="The name of judge model, which serve as a scorer to score the responses to evaluate the safety. gpt4 a.k.a gpt-4-0613 is the most expensive. Use default to keep align with related works."
    )
    parser.add_argument(
        "--input_dir",
        default=f"{home_dir}/data",
        type=str,
        help="The directory that stores the input data of LLM."
    )
    parser.add_argument(
        "--output_dir",
        default=f"{home_dir}/results/save_dir_placeholder",
        type=str,
        help="LLM's output directory."
    )
    parser.add_argument(
        "--input_dataset",
        default="advbench",
        type=str,
        choices = ['advbench', 'advbench-custom', 'hexphi-sub10', 'hexphi', 'jbb'],
        help="The benchmark dataset.",
    )
    parser.add_argument("--num_samples", type=int, default=-1, help="Only use partial number of samples in input_dataset. This used for test code is implemented correctly.")
    parser.add_argument('--num_processes', type=int, default=1) 
    parser.add_argument("--mode", type=int, default=2, help="mode: bit mask (high to low) bit1-run inference, bit2-eval ss, bit3-eval llm; 8 is to enforce rerun GPT to score again, i.e. refresh the GPTjudge's score.")
    parser.add_argument("--ps", type=str, default="base", help="prompt setting, 'ps' means 'prompt setting': base, mask-gpt, vitc-v/h-fontname, gcg-transfer.")   
    parser.add_argument("--defense", default=None, type=str, help="defense setting, default None means no defense.", choices=["ppl", "para", "retok", "self-reminder", "rpo"])
    parser.add_argument("--num_retries", default=1, type=int, help="The max number of times of retries.")
    parser.add_argument( "--debug", default=False, action="store_true", help="This flag controls whether run in debug manner.",)
    parser.add_argument("--num_selections", default=10, type=int, help="The number of random selected commendatory words in random_selection.")
    
    # Generation config of Attack LLM model in method.
    parser.add_argument(
        "--model_name",
        "--amodel",
        default="gpt-3.5-turbo-1106",
        type=str,
        help="The name of chosen LLM model to use in our method, a.k.a. amodel, attack model, i.e. use this LLM to launch the proposed jailbreak attack.",
    )
    # parser.add_argument(
    #     "--max_tokens",
    #     default=2048,
    #     type=int,
    #     help="The max_tokens parameter of chosen LLM model.",
    # )
    parser.add_argument(
        "--temperature",
        default=0.3,
        type=float,
        help="The temprature of the chosen LLM model.",
    )
    parser.add_argument(
        "--top_p",
        default=0.9,
        type=float,
        help="The `top_p` parameter of the chosen LLM model. Since `We generally recommend altering this or top_p but not both.`, we will not use it in GPT.",
    )
    parser.add_argument(
        "--frequency_penalty",
        default=0,
        type=float,
        help="The `frequency_penalty` parameter of the chosen LLM model.",
    )
    parser.add_argument(
        "--presence_penalty",
        default=0,
        type=float,
        help="The `presence_penalty` parameter of the chosen LLM model.",
    )
    parser.add_argument(
        "--stop_sequences",
        default=["\n\n"],
        nargs="+",
        help="The `stop_sequences` parameter of the chosen LLM model.",
    )
    parser.add_argument(
        "--logprobs",
        default=5,
        type=int,
        help="The `logprobs` parameter of the chosen LLM model"
    )
    parser.add_argument(
        "--n",
        default=1,
        type=int,
        help="The `n` parameter of the chosen LLM model. The number of responses to generate."
    )
    parser.add_argument(
        "--best_of",
        default=1,
        type=int,
        help="The `best_of` parameter of the chosen LLM model. The beam size on the the chosen LLM model server."
    )
    
    # Logger config
    mdHMS = datetime.now().strftime("%m_%d-%H_%M_%S")
    parser.add_argument("--dump_path", type=str, default=f"{home_dir}/dump-logs/", help="Experiment dump path")
    parser.add_argument("--exp_name", type=str, default="exp_name_placeholder", help="Experiment name")
    parser.add_argument("--exp_id", type=str, default=f"{mdHMS}", help="Experiment ID")

    
    return parser.parse_args()





if __name__ == '__main__':
    
    args = parse_args()
    args_dict = vars(args)
    max_key_length = max(len(str(key)) for key in args_dict)
        
    for key in sorted(args_dict):
        print(f"{str(key):<{max_key_length}} : {args_dict[key]}")


    # for key in args_dict:
    #     print(f"{key}: {args_dict[key]}")
    # for key in sorted(args_dict):
    #     print(f"{key}: {args_dict[key]}")