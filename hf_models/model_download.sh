# victim models: dLLMs
huggingface-cli download GSAI-ML/LLaDA-8B-Instruct --local-dir ./LLaDA-8B-Instruct --resume-download
huggingface-cli download GSAI-ML/LLaDA-1.5 --local-dir ./LLaDA-1.5 --resume-download
huggingface-cli download Dream-org/Dream-v0-Instruct-7B --local-dir ./Dream-v0-Instruct-7B --resume-download
huggingface-cli download Gen-Verse/MMaDA-8B-MixCoT --local-dir ./MMaDA-8B-MixCoT --resume-download

huggingface-cli download Dream-org/Dream-Coder-v0-Instruct-7B --local-dir ./Dream-Coder-v0-Instruct-7B --resume-download
huggingface-cli download Dream-org/DreamOn-v0-7B --local-dir ./DreamOn-v0-7B --resume-download
huggingface-cli download apple/DiffuCoder-7B-Instruct --local-dir ./DiffuCoder-7B-Instruct --resume-download
huggingface-cli download apple/DiffuCoder-7B-cpGRPO --local-dir ./DiffuCoder-7B-cpGRPO --resume-download

# evaluator--ASR-e
huggingface-cli download cais/HarmBench-Llama-2-13b-cls --local-dir ./HarmBench-Llama-2-13b-cls --resume-download
huggingface-cli download qylu4156/strongreject-15k-v1 --local-dir ./strongreject-15k-v1 --resume-download