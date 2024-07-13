from vllm import LLM, SamplingParams
import pdb
import sys
import numpy as np
sys.path.append('.')
from common.registry import registry
from prompts.prompt_template import prompt_templates
from typing import Optional, Union, Literal
import torch
import re
import copy
from openai import OpenAI
import time
import openai

@registry.register_llm("openai_vllm")
class OPENAI_VLLM:
    def __init__(self,
                 model,
                 temperature=0,
                 max_tokens=100,
                 system_message="You are a helpful assistant.",
                 use_azure=True,
                 top_p=1,
                 stop='\n',
                 retry_delays=5, # in seconds
                 max_retry_iters=5,
                 context_length=4096,
                 ):
        
        
        self.engine =  model
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.system_message = system_message
        self.use_azure = use_azure
        self.top_p = top_p
        self.stop = stop
        self.retry_delays = retry_delays
        self.max_retry_iters = max_retry_iters
        self.context_length = context_length
        
        self.client = OpenAI(
            api_key = "EMPTY",
            base_url = "http://localhost:8000/v1"
            # organization='',
        )
                
    def chat_inference(self, messages):

        response = openai.ChatCompletion.create(
            engine=self.engine, # engine = "deployment_name".
            messages=messages,
            stop = self.stop,
            temperature = self.temperature,
            max_tokens = self.max_tokens,
        )

        return response['choices'][0]['message']['content']
    
    def chat_inference_with_config(self, messages, config):
        stop = config.get("stop", self.stop)
        temperature = config.get("temperature", self.temperature)
        max_tokens = config.get("max_tokens", self.max_tokens)
        n = config.get("n", 1)
        
        stop = config.get("stop", self.stop)
        temperature = config.get("temperature", self.temperature)
        max_tokens = config.get("max_tokens", self.max_tokens)
        n = config.get("n", 1)
        logprobs = config.get("logprobs", 5)
        do_sample = config.get("do_sample", True)
        top_p = config.get("top_p", 1)
        
        
        response = openai.ChatCompletion.create(
                        model=self.engine,
                        messages=messages,
                        max_tokens=max_tokens,
                        temperature=temperature,
                        top_p=top_p,
                        n=n,
                        stop=stop,
                        logprobs=logprobs,
        )
        if logprobs == 0:
            if n == 1:
                return response.choices[0].message.content
            else:
                return [choice.message.content for choice in response.choices]
        else:
            
            # return a list of {"text":xxx, "logprobs":xxx, "tokens": xxx}
            
            outputs = []
            for i in range(n):
                choice = response.choices[i]
                item = {}
                item["text"] = choice.message.content
                raw_log_prob = choice.logprobs
                item["logprobs"] = [token.logprob for token in raw_log_prob]
                item["tokens"] = [token.token for token in raw_log_prob]
                outputs.append(item)
            
            if n == 1:
                return outputs[0]
            else:
                return outputs
    
    def generate(self, system_message, prompt):
        prompt=[
                {"role": "system", "content": system_message},
                {"role": "user", "content": prompt}
        ]
        for attempt in range(self.max_retry_iters):  
            try:
                output = self.chat_inference(prompt)
                # output = output.split("\n")[0]
                return True, output # return success, completion
            except Exception as e:
                print(f"Error on attempt {attempt + 1}") 
                if attempt < self.max_retry_iters - 1:  # If not the last attempt
                    time.sleep(self.retry_delays)  # Wait before retrying
                
                else:
                    print("Failed to get completion after multiple attempts.")
                    # raise e
                    
        return False, None
    
    def generate_with_config(self, system_message, prompt, config):
        prompt=[
                {"role": "system", "content": system_message},
                {"role": "user", "content": prompt}
        ]
        for attempt in range(self.max_retry_iters):  
            try:
                output = self.chat_inference_with_config(prompt, config)
                # output = output.split("\n")[0]
                return True, output # return success, completion
            except Exception as e:
                print(f"Error on attempt {attempt + 1}") 
                if attempt < self.max_retry_iters - 1:  # If not the last attempt
                    time.sleep(self.retry_delays)  # Wait before retrying
                
                else:
                    print("Failed to get completion after multiple attempts.")
                    # raise e
        return False, None
    
    
    def num_tokens_from_messages(self, messages):
        raise NotImplementedError
        return 1

    @classmethod
    def from_config(cls, config):

        engine = config.get("engine", "gpt-35-turbo")
        temperature = config.get("temperature", 0)
        max_tokens = config.get("max_tokens", 100)
        top_p = config.get("top_p", 1)
        stop = config.get("stop", ["\n"])
        context_length = config.get("context_length", 4096)
        retry_delays = config.get("retry_delays", 5)
        max_retry_iters = config.get("max_retry_iters", 5)
        system_message = config.get("system_message", "You are a helpful assistant.")

        return cls(
            engine,
            temperature=temperature,
            max_tokens=max_tokens,
            top_p=top_p,
            stop=stop,
            context_length=context_length,
            retry_delays=retry_delays,
            max_retry_iters=max_retry_iters,
            system_message=system_message
        )