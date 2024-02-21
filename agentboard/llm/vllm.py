from vllm import LLM, SamplingParams
import pdb
import sys
sys.path.append('.')
from common.registry import registry
from prompts.prompt_template import prompt_templates

import torch


@registry.register_llm("vllm")
class VLLM:
    def __init__(self,
                 model='',
                 temperature=0,
                 max_tokens=100,
                 top_p=1.0,
                 context_length=4096,
                 stop='\n',
                 ngpu=4,
                 d_type='bfloat16',
                 tau=None
                 ):
        self.engine = model
        self.model = model
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.top_p = top_p
        self.stop = stop
        self.context_length = context_length
        if self.context_length > 8192:
            self.llm = LLM(model=self.model, dtype=d_type, tensor_parallel_size=ngpu, gpu_memory_utilization=0.9, max_num_batched_tokens=8192, max_model_len=8192)
        else:
            self.llm = LLM(model=self.model, dtype=d_type, tensor_parallel_size=ngpu, gpu_memory_utilization=0.8, max_num_batched_tokens=self.context_length)
        
        self.tokenizer = self.llm.get_tokenizer()
        
        if tau is None:
            self.sampling_params = SamplingParams(
                temperature=temperature,
                top_p=top_p,
                stop=stop,
                max_tokens=max_tokens,
            )
        else:
            self.sampling_params = SamplingParams(
                temperature=temperature,
                top_p=top_p,
                stop=stop,
                max_tokens=max_tokens,
                logits_processors=[
                    UncertaintyLogitsProcessor(self.tokenizer, tau)
                ]
            )
        
        
    def make_prompt(self, system_message, prompt):
        full_prompt = None
        system_message += "Generate your next step of action after Action. Action must not be empty. e.g. Action: put down cup. \n"
        if "codellama-13b" in self.model.lower():
            full_prompt = prompt_templates["codellama-13b"].format(system_prompt=system_message, prompt=prompt)
            full_prompt = full_prompt.strip()
        elif "codellama-34b" in self.model.lower():
            full_prompt = prompt_templates["codellama-34b"].format(system_prompt=system_message, prompt=prompt)
            full_prompt = full_prompt.strip()
            
        elif "llama" in self.model.lower():
            full_prompt = prompt_templates["llama"].format(system_prompt=system_message, prompt=prompt)
            full_prompt = full_prompt.strip()
            
        elif 'lemur' in self.model.lower():
            full_prompt = prompt_templates["lemur"].format(system_prompt=system_message, prompt=prompt)
            full_prompt = full_prompt.strip()
            
        elif 'vicuna' in self.model.lower():
            full_prompt = prompt_templates["vicuna"].format(system_prompt=system_message, prompt=prompt)
            full_prompt = full_prompt.strip()
            
        elif 'deepseek' in self.model.lower():
            full_prompt = prompt_templates["deepseek"].format(system_prompt=system_message, prompt=prompt)
            full_prompt = full_prompt.strip()
        
        elif 'mistral' in self.model.lower():
            full_prompt = prompt_templates["mistral"].format(system_prompt=system_message, prompt=prompt)
            full_prompt = full_prompt.strip()
            
        else:
            raise NotImplementedError
        
        return full_prompt

    def generate(self, system_message, prompt):
        full_prompt = self.make_prompt(system_message, prompt)
        assert full_prompt is not None
        outputs = self.llm.generate([full_prompt], self.sampling_params)
        outputs = outputs[0].outputs[0].text
        
        if 'vicuna' in self.model.lower():
            # Note: vicuna tends to generate get\_search\_movie with Action Input: {"movie\_name": "Crouching Tiger, Hidden Dragon"} when using tools
            outputs = outputs.replace('\_', '_')

        return True, outputs 
    
    def generate_uncertainty_early_stop(self, system_message, prompt):
        
        full_prompt = self.make_prompt(system_message, prompt)
        assert full_prompt is not None
        outputs = self.llm.generate([full_prompt], self.sampling_params)
        outputs = outputs[0].outputs[0].text
        
        if 'vicuna' in self.model.lower():
            # Note: vicuna tends to generate get\_search\_movie with Action Input: {"movie\_name": "Crouching Tiger, Hidden Dragon"} when using tools
            outputs = outputs.replace('\_', '_')

        return True, outputs 

    def num_tokens_from_messages(self, messages):
        prompt = messages[1]["content"]
        system_message = messages[0]["content"]
        full_prompt = self.make_prompt(system_message, prompt)
        tokens = self.tokenizer(full_prompt)
        num_tokens = len(tokens["input_ids"])
        #print(num_tokens)
        return num_tokens

    @classmethod
    def from_config(cls, config):

        engine = config.get("engine", "gpt-35-turbo")
        temperature = config.get("temperature", 0)
        max_tokens = config.get("max_tokens", 100)
        top_p = config.get("top_p", 1)
        stop = config.get("stop", ["\n"])
        context_length = config.get("context_length", 4096)
        ngpu = config.get("ngpu", 4)
        dtype = config.get("dtype", 'bfloat16')
        tau = config.get("tau", None)
        return cls(model=engine,
                   temperature=temperature,
                   max_tokens=max_tokens,
                   top_p=top_p,
                   context_length=context_length,
                   stop=stop,
                   ngpu=ngpu,
                   d_type=dtype,
                   tau=tau)



class UncertaintyLogitsProcessor:
    def __init__(
            self, 
            tokenizer,
            tau=0.1,
            random_interrupt=False
        ):
            self.tokenizer = tokenizer
            self.tau = tau
            self.random_interrupt = random_interrupt
            
            self.eos_id = tokenizer.eos_token_id

    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor) -> torch.FloatTensor:
        max_prob = torch.softmax(scores, dim=-1).amax(dim=-1)
        _mask = max_prob < self.tau # [B]
        if self.random_interrupt:
            # re-scale the prob such that
            # prob > self.tau => sample_prob = 0.0
            # prob = self.tau => sample_prob = 0.5
            # prob = 0.0 => sample_prob = 1.0
            calibrated_prob = (
                _mask.float() * (1 - (0.5 * max_prob / self.tau))
            )
            uncertain_mask = torch.bernoulli(calibrated_prob).bool()
        else:
            uncertain_mask = _mask # [B]
        
        
        ###########################################
        # emit an <eos> token if 
        force_eos_mask = uncertain_mask
        
        # produce a point mass over <eos>
        scores[force_eos_mask, :] = -float("inf")
        scores[force_eos_mask, self.eos_id] = 0
        
        return scores