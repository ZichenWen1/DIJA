from typing import Any
from openai import OpenAI
import google.generativeai as genai
import boto3
import json
import time
import pdb
import os
import logging
from openai import AzureOpenAI
import anthropic

from typing import Any
import requests
import json
import os
import logging
from colorama import Fore
import time

gpt_apis = ['PLACEHOLDER_FOR_YOUR_API_KEY', ]
claude_apis = ['PLACEHOLDER_FOR_YOUR_API_KEY',]

AZURE_OPENAI_ENDPOINT = "PLACEHOLDER_FOR_YOUR_AZURE_OPENAI_ENDPOINT"
AZURE_OPENAI_API_KEY = "PLACEHOLDER_FOR_YOUR_AZURE_OPENAI_API_KEY"
os.environ["AZURE_OPENAI_ENDPOINT"] = AZURE_OPENAI_ENDPOINT
os.environ["AZURE_OPENAI_API_KEY"] = AZURE_OPENAI_API_KEY

logger = logging.getLogger()
logging.getLogger("urllib3").setLevel(logging.WARNING)
logging.getLogger("requests").setLevel(logging.WARNING)

from colorama import Fore

class GPT:
    
    def __init__(self, model_name, temperature=0.3, seed=None, api_idx=0):           #* api_idx is the index of gpt_apis or claude_apis
        
        self.model_name = model_name
        self.client = OpenAI(api_key=gpt_apis[api_idx])
        self.T = temperature
        self.seed=seed
    
    def __call__(self, prompt, n:int=1, debug=False, **kwargs: Any) -> Any:    #* n: How many chat completion choices to generate for each input message.
        
        if debug:
            message = [{'role':'user', 'content':prompt}]
            response = self.client.chat.completions.create(messages=message, n=n, model=self.model_name, temperature=self.T, seed=self.seed, **kwargs)
            return response
        else:
            return self.eval_call_wrapper(prompt, n, self.model_name, self.T, self.seed, **kwargs)


    def eval_call(self, prompt, n:int=1, debug=False, **kwargs: Any) -> Any:
        if debug:
            message = [{'role':'user', 'content':prompt}]
            response = self.client.chat.completions.create(messages=message, n=n, model=self.model_name, temperature=self.T, seed=self.seed, **kwargs)
            return response
        else:
            return self.eval_call_wrapper(prompt, n, self.model_name, self.T, self.seed, **kwargs)
        
    def eval_call_wrapper(self, prompt, n, model, temperature, seed, **kwargs: Any) -> Any:
        retry_count, retries = 0, 5
        response = None
        while retry_count < retries:
            try:
                message = [{'role':'user', 'content':prompt}] if 'system' not in kwargs else [{'role':'system', 'content':kwargs['system']}, {'role':'user', 'content':prompt}]
                # print(Fore.RED + f"message: {message}" + Fore.RESET)
                if 'system' in kwargs: kwargs.pop('system')
                
                if 'o1' not in self.model_name and 'o3-mini' not in self.model_name:
                    response = self.client.chat.completions.create(messages=message, n=n, model=model, temperature=temperature, seed=seed, max_tokens=2048, **kwargs)
                else:
                    if 'o1-mini' in self.model_name:
                        response = self.client.chat.completions.create(messages=message, n=n, model=model, seed=seed, max_completion_tokens=2048, **kwargs)
                    else:
                        response = self.client.chat.completions.create(messages=message, n=n, model=model, seed=seed, max_completion_tokens=2048, reasoning_effort='medium', **kwargs)
                break
            except Exception as e:
                retry_count += 1
                backoff_time = 2**retry_count+1
                time.sleep(backoff_time)
                logger.exception(f"error exception: {e}")
                logger.info(f"Retrying in {backoff_time} seconds...")     
        if response == None : logger.error(f'>>> Fail to generate response after trying {retries} times invoking. The response will be initial `None`.')
        
        return response
    
    def resp_parse(self, response)->list:
        n = len(response.choices)
        return [response.choices[i].message.content for i in range(n)]


class AzureGPT4o:
    
    def __init__(self, model_name, temperature=0.3, seed=None, api_idx=0):
        
        self.model_name = model_name
        self.client = AzureOpenAI(azure_endpoint = os.getenv("AZURE_OPENAI_ENDPOINT"), api_key=os.getenv("AZURE_OPENAI_API_KEY"),  api_version="2024-02-01")
        self.T = temperature
        self.seed=seed
    
    def __call__(self, prompt, n:int=1, debug=False, **kwargs: Any) -> Any:    #* n: How many chat completion choices to generate for each input message.
        
        if debug:
            message = [{'role':'user', 'content':prompt}]
            response = self.client.chat.completions.create(messages=message, n=n, model="spdzcoder", temperature=self.T, seed=self.seed, **kwargs)
            return response
        else:
            return self.eval_call_wrapper(prompt, n, "spdzcoder", self.T, self.seed, **kwargs)


    def eval_call(self, prompt, n:int=1, debug=False, **kwargs: Any) -> Any:
        if debug:
            message = [{'role':'user', 'content':prompt}]
            response = self.client.chat.completions.create(messages=message, n=n, model="spdzcoder", temperature=self.T, seed=self.seed, **kwargs)
            return response
        else:
            return self.eval_call_wrapper(prompt, n, "spdzcoder", self.T, self.seed, **kwargs)
        
    def eval_call_wrapper(self, prompt, n, model, temperature, seed, **kwargs: Any) -> Any:
        retry_count, retries = 0, 5
        response = None
        while retry_count < retries:
            try:
                message = [{'role':'user', 'content':prompt}]
                response = self.client.chat.completions.create(messages=message, n=n, model=model, temperature=temperature, seed=seed, max_tokens=2048, **kwargs)
                break
            except Exception as e:
                retry_count += 1
                backoff_time = 2**retry_count+1
                time.sleep(backoff_time)
                logger.exception(f"error exception: {e}")
                logger.info(f"Retrying in {backoff_time} seconds...")     
        if response == None : logger.error(f'>>> Fail to generate response after trying {retries} times invoking. The response will be initial `None`.')
        
        return response
    
    def resp_parse(self, response)->list:
        n = len(response.choices)
        return [response.choices[i].message.content for i in range(n)]


class AnthropicClaude:
    def __init__(self, model_name, temperature=0.3, api_idx=0) -> None:
        self.T = temperature
        if model_name == 'claude-v2':
            self.model_name = 'claude-2.0'                  # $8/$24
        elif model_name == 'claude-v2.1':
            self.model_name = 'claude-2.1'                  # $8/$24, an updated version of claude-2.0
        elif model_name == 'claude-3-opus':
            self.model_name = 'claude-3-opus-20240229'      # $15/$75
        elif model_name == 'claude-3-sonnet':
            self.model_name = 'claude-3-sonnet-20240229'    # $3/$15
        elif model_name == 'claude-3-haiku':
            self.model_name = 'claude-3-haiku-20240307'     # $0.25/$1.25
        elif model_name == 'claude-3.5-sonnet':
            self.model_name = 'claude-3-5-sonnet-20240620'  # $3/$15 # claude-3-5-sonnet-20241022
        elif model_name == 'claude-3.5-haiku':
            self.model_name = 'claude-3-5-haiku-20241022'   # $1/$5
        else:
            raise ValueError(f"model_name should be in https://docs.anthropic.com/en/docs/about-claude/models, but got {model_name}")
        self.client = anthropic.Anthropic(api_key = claude_apis[api_idx])
    
    def __call__(self, prompt, n=1, debug=False, **kwargs: Any) -> Any:

        if debug:
            message = [{"role":"user", "content":prompt}]
            response = self.client.messages.create(messages=message, model=self.model_name, temperature=self.T, max_tokens=2048, **kwargs) # system="You are helpful assistant.",stop=your_stop_sequences
            return response
        else:
            return self.call_wrapper(prompt, self.model_name, self.T, **kwargs)

    def call_wrapper(self, prompt, model, temperature, **kwargs: Any) -> Any:
        retry_count, retries = 0, 5
        response = None
        while retry_count < retries:
            try:
                message = [{"role":"user", "content":prompt}]
                # if 'system' in kwargs: 
                #     print(Fore.RED + f"system_prompt: {kwargs['system']}" + Fore.RESET) 
                response = self.client.messages.create(messages=message, model=model, temperature=temperature, max_tokens=2048, **kwargs) # system="You are helpful assistant.",stop=your_stop_sequences
                break
            except Exception as e:
                retry_count += 1
                backoff_time = 2**retry_count+1
                time.sleep(backoff_time)
                logger.exception(f"error exception: {e}")
                logger.info(f"Retrying in {backoff_time} seconds...")     
        if response == None : logger.error(f'>>> Fail to generate response after trying {retries} times invoking. The response will be initial `None`.')

        return response

    def resp_parse(self, response) -> list:
        n = len(response.content)
        return [response.content[i].text for i in range(n)]


class Llama3_api:
    def __init__(self, model_name, temperature=0.6) -> None:
        assert 'llama3-api' in model_name
        self.T = temperature        
        self.client = OpenAI(api_key="PLACEHOLDER_FOR_YOUR_DEEPINFRA_API_KEY", base_url="https://api.deepinfra.com/v1/openai")
        # self.client = OpenAI(api_key="PLACEHOLDER_FOR_YOUR_SILICONFLOW_API_KEY", base_url="https://api.siliconflow.cn/v1")
  
        if '8b' in model_name:
            self.model_DI = "meta-llama/Meta-Llama-3-8B-Instruct"
        elif '70b' in model_name:
            self.model_DI = "meta-llama/Meta-Llama-3-70B-Instruct"
        else:
            raise ValueError(f"model_name should be llama3-api-7b or llama3-api-70b, but got {model_name}")
        
    def __call__(self, prompt, n=1, debug=False, top_p=0.9, **kwargs: Any) -> Any:
        prompt = [{'role':'user', 'content':prompt}] if 'system' not in kwargs else [{'role':'system', 'content':kwargs['system']}, {'role':'user', 'content':prompt}] 
        # print(Fore.RED + f"message: {prompt}" + Fore.RESET)
        if debug:
            response = self.client.chat.completions.create(messages=prompt, n=n, model=self.model_DI, temperature=self.T, top_p=top_p, **kwargs)
        else:
            response = self.call_wrapper(messages=prompt, n=n, model=self.model_DI, temperature=self.T, top_p=top_p, **kwargs)
        return response
    

    def call_wrapper(self, **kwargs):
        retry_count, retries = 0, 5
        response = None
        while retry_count < retries:
            try:
                # print(Fore.BLUE + f"kwargs: {kwargs}" + Fore.RESET)
                if 'system' in kwargs: kwargs.pop('system') 
                response = self.client.chat.completions.create(**kwargs)
                break
            except Exception as e:
                retry_count += 1
                backoff_time = 2**retry_count+1
                time.sleep(backoff_time)
                logger.exception(f"error exception: {e}")
                logger.info(f"Retrying in {backoff_time} seconds...")     
        if response == None : logger.error(f'>>> Fail to generate response after trying {retries} times invoking. The response will be initial `None`.')        
        
        return response

    def resp_parse(self, response)->list:
        n = len(response.choices)
        return [response.choices[i].message.content for i in range(n)]


class Deepseek_api:
    def __init__(self, model_name, temperature=0.6) -> None:
        assert 'deepseek' in model_name
        self.T = temperature
        self.api_key = "PLACEHOLDER_FOR_YOUR_SILICONFLOW_API_KEY"
        self.client = OpenAI(api_key=self.api_key, base_url="https://api.siliconflow.cn/v1")
        
        if 'r1' in model_name:
            self.model_DI = "Pro/deepseek-ai/DeepSeek-R1"
        elif 'v3' in model_name:
            self.model_DI = "deepseek-ai/DeepSeek-V3"
        else:
            raise ValueError(f"Model_name should start with deepseek-r1 or deepseek-v3, but got {model_name}.")
        
    def __call__(self, prompt, n=1, debug=False, top_p=0.9, **kwargs: Any) -> Any:
        prompt = [{'role':'user', 'content':prompt}] if 'system' not in kwargs else [{'role':'system', 'content':kwargs['system']}, {'role':'user', 'content':prompt}]
        # print(Fore.RED + f"message: {prompt}" + Fore.RESET)
        if debug:
            response = self.client.chat.completions.create(messages=prompt, n=n, model=self.model_DI, temperature=self.T, top_p=top_p, **kwargs)
        else:
            response = self.call_wrapper(messages=prompt, n=n, model=self.model_DI, temperature=self.T, top_p=top_p, **kwargs)
        return response
    

    def call_wrapper(self, **kwargs):
        retry_count, retries = 0, 5
        response = None
        while retry_count < retries:
            try:
                # print(Fore.BLUE + f"kwargs: {kwargs}" + Fore.RESET)
                if 'system' in kwargs: kwargs.pop('system')
                response = self.client.chat.completions.create(**kwargs)
                break
            except Exception as e:
                retry_count += 1
                backoff_time = 2**retry_count+1
                time.sleep(backoff_time)
                logger.exception(f"error exception: {e}")
                logger.info(f"Retrying in {backoff_time} seconds...")     
        if response == None : logger.error(f'>>> Fail to generate response after trying {retries} times invoking. The response will be initial `None`.')        
        
        return response

    def resp_parse(self, response)->list:
        n = len(response.choices)
        return [response.choices[i].message.content for i in range(n)]



class CustomGPT:
    def __init__(self, model_name, base_url="http://35.220.164.252:3888/v1", api_key="", temperature=1.0):
        self.model_name = model_name
        self.base_url = base_url
        self.api_key = api_key
        self.temperature = temperature
        self.client = OpenAI(base_url=self.base_url, api_key=self.api_key)

    def __call__(self, prompt, n=1, debug=False, **kwargs: Any) -> Any:
        if debug:
            return self.chat_completion(prompt, n=n, model=self.model_name, temperature=self.temperature, **kwargs)
        else:
            return self.eval_call_wrapper(prompt, n=n, model=self.model_name, temperature=self.temperature, **kwargs)

    def chat_completion(self, prompt, n=1, model="gpt-4o", temperature=1.0, **kwargs):
        headers = {
            'Authorization': f'Bearer {self.api_key}',
            'Content-Type': 'application/json'
        }
        # pdb.set_trace()
        data = {
            "model": model,
            "messages": [
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": prompt}
            ],
            "temperature": temperature,
            "n": n,
            **kwargs
        }
        response = requests.post(f"{self.base_url}/chat/completions", headers=headers, json=data)
        return response.json()

    def eval_call_wrapper(self, prompt, n=1, model="gpt-4o", temperature=1.0, **kwargs):
        retry_count, retries = 0, 5
        response = None
        while retry_count < retries:
            try:
                message = [{"role": "user", "content": prompt}]
                response = self.client.chat.completions.create(
                    model=model,
                    messages=message,
                    temperature=temperature,
                    n=n,
                    **kwargs
                )
                break
            except Exception as e:
                retry_count += 1
                backoff_time = 2**retry_count + 1
                time.sleep(backoff_time)
                logger.exception(f"error exception: {e}")
                logger.info(f"Retrying in {backoff_time} seconds...")

        if response is None:
            logger.error(f'>>> Fail to generate response after trying {retries} times invoking.')
        return response

    def resp_parse(self, response) -> list:
        if isinstance(response, dict):  # 处理 requests 返回的 JSON
            return [choice['message']['content'].strip() for choice in response.get('choices', [])]
        elif hasattr(response, 'choices'):  # 处理 OpenAI SDK 返回对象
            return [choice.message.content.strip() for choice in response.choices]
        return []


def load_model(model_name, **kwargs):
    
    # if model_name == "azure-gpt4o":
    #     return AzureGPT4o(model_name, **kwargs)
    
    # elif "gpt" in model_name and "gpt2" not in model_name:
    #     return GPT(model_name, **kwargs)
    
    # elif "claude" in model_name:
    #     return AnthropicClaude(model_name, **kwargs)
    
    # elif "llama3" in model_name:
    #     return Llama3_api(model_name, **kwargs)

    # elif "deepseek" in model_name:
    #     return Deepseek_api(model_name, **kwargs)
    
    # elif "o1" in model_name or "o3-mini" in model_name:
    #     return GPT(model_name, **kwargs)
    
    # else:
    #     raise ValueError(f"model_name invalid")
    return CustomGPT(model_name, **kwargs)


