import pdb

from common.registry import registry
# from rouge import Rouge
import json
import random
import re


@registry.register_algorithm("Generation")
class Generation:  # the agent should receive goal, state and action, then return the next state
    def __init__(self,
                 llm_model,
                 prompt_path
                 ):
        
        self.llm_model = llm_model
        
        self.prompts = json.load(open(prompt_path, 'r'))
    
    def make_prompt(self, question, prompt):
        query = ""
        query += prompt["instruction"] + '\n'
        query += "\nHere is an example:\n" + prompt["examples"][0] + '\n'
        
        input_prompt = query + question
        input_prompt = input_prompt + "\nAnswer:"

        return input_prompt

    def parse_integer_lists(self, string):
        pattern = r'\[([\d\s,]+)\]'  # Regular expression pattern to match integer lists
        matches = re.findall(pattern, string)  # Find all matches

        # Parse each match
        result = []
        for match in matches:
            integers = re.findall(r'\d+', match)  # Find all integers in the match
            integers = [int(num) for num in integers]  # Convert strings to integers
            result.append(integers)

        return result
    def run(self, question):
        
        system_message = self.prompts["system_msg"]
        input_prompt = self.make_prompt(question, self.prompts)
        success, results = self.llm_model.generate( system_message, input_prompt)
        
        if success:
            result_lists = self.parse_integer_lists(results)
            if len(result_lists) > 0:
                return True, result_lists[-1]
        return False, None

    @classmethod
    def from_config(cls, llm_model, config):
        return cls(llm_model, config["prompt_path"])