import sys
import os
import re
import warnings
import yaml
import json
import time
import math
import datasets
import argparse
import timeout_decorator
from dotenv import load_dotenv
from llm import load_llm
from algorithms import load_algorithm
from tqdm import tqdm
from typing import Optional

from utils.math.math_utils import parse_question, parse_ground_truth, math_equal, call_with_timeout

os.environ["CUDA_VISIBLE_DEVICES"] = "3"

def load_analyze_data(task, path='result/...'):
    examples = []
    with open(path, "r") as f:
        for line in f:
            js = json.loads(line)
            examples.append(js)
    return examples
  

class Iterator:
    def __init__(self, data, step=200):
        self.data = data
        self.index = 0
        self.step = step

    def __iter__(self):
        return self

    def __next__(self):
        if self.index >= len(self.data):
            raise StopIteration
        max_index = min(self.index+self.step, len(self.data))
        value = self.data[self.index:max_index]
        self.index += self.step
        return value

   
class AnalyzeLogp:
    def __init__(self,
                 task="dp",
                 run_config=None,
                 llm_config=None,
                 algorithm_config=None,
                 ):
        
        self.llm = load_llm(llm_config.get("name", "gpt"), llm_config)
        
        self.log_path = run_config["log_path"]
        self.task = task
        
        self.batch_size = run_config["batch_size"]
        self.evaluator = load_algorithm(algorithm_config["name"], algorithm_config, self.llm)
        
        self.examples = load_analyze_data(task, run_config["data_path"])
        
    def parallel_evaluate(self):
        
        # create a new empty file for logging
        f = open(os.path.join(self.log_path,f"{self.task}_logp_values.jsonl"), "w")
        
        item_iter = Iterator(self.examples, self.batch_size)
        
        id = 0
        
        for test_items in tqdm(item_iter, total=math.ceil(len(self.examples)/self.batch_size)):
            prompts = [item["prompt"] for item in test_items]
            completions = [item["completion"] for item in test_items]
            
            if self.task == "math":
                system_msg = "You will write python program to solve math problems. You will only write imports and code blocks ."
            elif self.task == "gsm8k":
                system_msg = "You will write python program to solve math problems. You will only write code blocks."
            else:
                system_msg = "Finish writing the python function. You will only write code blocks."
                
            success, all_values = self.evaluator.parallel_run(system_msg=system_msg, prompts=self.prompts, completions=completions) # process all questions in parallel

            for i, item in enumerate(test_items):
                item["logp"] = all_values[i]
                f.write(json.dumps(item) + "\n") # write the logp values to the file
    
def parse_args():
    parser = argparse.ArgumentParser(description="Testing")

    parser.add_argument("--cfg-path", required=True, help="path to configuration file.")
    parser.add_argument("--tasks", required=True, type=str, help="specify the tasks")
    parser.add_argument("--algorithm", required=True, type=str, help="specify the algorithm")
    parser.add_argument("--model", required=True ,help="specify the models, available models are stated in the configuration file")
    parser.add_argument("--log_path", required=False, default='', help="specify the place to store the resuls")
    parser.add_argument("--data_path", required=False, default='/root/huggingface/gsm8k', help="specify the test data file")
    parser.add_argument("--batch_size", required=False, default=50, type=int,help="number of problems processed together")
    # parser.add_argument("--prompt_path", required=False, default='', help="specify the prompt")
    args = parser.parse_args()

    return args


def path_constructor(loader, node):
    path_matcher = re.compile(r'\$\{([^}^{]+)\}')
    ''' Extract the matched value, expand env variable, and replace the match '''
    value = node.value
    match = path_matcher.match(value)
    env_var = match.group()[2:-1]
    return os.environ.get(env_var) + value[match.end():]

def load_config(cfg_path, args):
    path_matcher = re.compile(r'\$\{([^}^{]+)\}')
    yaml.add_implicit_resolver('!path', path_matcher)
    yaml.add_constructor('!path', path_constructor)
    with open(cfg_path, "r") as f:
        config = yaml.load(f, Loader=yaml.FullLoader)
    llm_config = config["llm"]
    algorithm_config = config["algorithm"]
    run_config = config["run"]
    algorithm_config["name"] = args.algorithm
    if args.log_path != '':
        run_config["log_path"] = args.log_path
    if args.data_path != '':
        run_config["data_path"] = args.data_path
    run_config["batch_size"] = args.batch_size
    # if args.prompt_path != '':
    #     algorithm_config["prompt_path"] = args.prompt_path
    return llm_config, algorithm_config, run_config

def check_log_paths_are_ready(log_dir):

    if not os.path.exists(log_dir):
        os.makedirs(log_dir)
            
    return True


def main():
    load_dotenv()  # take environment variables from .env., load openai api key, tool key, wandb key, project path...

    args = parse_args()
    
    llm_config, algorithm_config, run_config = load_config(args.cfg_path, args) 
    
    task = args.tasks
    llm_config = llm_config[args.model]
    
    check_log_paths_are_ready(run_config["log_path"])
    
    # save the configuration file
    with open(os.path.join(run_config["log_path"], "config.yaml"), "w") as f:
        yaml.dump({"llm":llm_config, "algorithm":algorithm_config, "run":run_config}, f)
    
    metrics_generation = AnalyzeLogp(task, run_config, llm_config, algorithm_config)
    
    metrics_generation.parallel_evaluate()
    
    
if __name__ == "__main__":
    main()