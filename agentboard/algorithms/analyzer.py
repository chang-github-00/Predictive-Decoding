import pdb

from common.registry import registry
# from rouge import Rouge
import json
import random
import re
import io
import argparse


@registry.register_algorithm("Logprob_Analyzer")

class Logprob_Analyzer:
    def __init__(self, 
                 llm_model, 
                 prompt_path=None):
        self.llm_model = llm_model
    
    def parallel_run(self, prompt, completion):
        # get logp of completion given prompt
        logp = self.llm_model.logprob(prompt, completion)
        return logp