import matplotlib.pyplot as plt
import numpy as np
import json
import os
import seaborn as sns
from scipy.stats import pearsonr

def load_analyze_data(task, path='result/...'):
    examples = []
    path = os.path.join(path)
    with open(path, "r") as f:
        for line in f:
            js = json.loads(line)
            examples.append(js)
    if task == "humaneval": examples = examples[:164]
    return examples

task = "gsm8k"
prob_data_path = f"results/run_anlaysis_important_results/{task}_corrected_output_probs_per_step.jsonl"
with open(prob_data_path, "r") as f:
    probs = json.load(f)

evaluate_all_probs = []
evaluate_all_gts = []

for i,prob in enumerate(probs):
    all_probs = [p[0] for p in prob]
    all_gts = [p[1] for p in prob]
    
    all_probs = [p for p in all_probs if np.isnan(p) == False]
    all_gts = all_gts[:len(all_probs)]
    evaluate_all_probs.extend(all_probs)
    evaluate_all_gts.extend(all_gts)
    
    if i not in [3,10,43,52,56]:
        continue
    if 0 in all_gts:


        first_false = max(all_gts.index(False)-1, 0)
        plt.figure(figsize=(4, 2))
        plt.plot(all_probs, label='Probability of Correct Answer', marker='o', markersize=3)
        # draw a line to indicate the first false
        plt.axvline(first_false, color='red', linestyle='--', label='First Incorrect Answer')
        plt.xlabel('Step')
        plt.ylabel('LLM Score') 
        plt.savefig(f"draw/{task}_example_prm_{i}.pdf", bbox_inches='tight', pad_inches=0.1,format="pdf")
    else:
        plt.figure(figsize=(4, 2))
        plt.plot(all_probs, label='Probability of Correct Answer', marker='o', markersize=3)
        plt.xlabel('Step')
        plt.ylabel('LLM Score') 
        plt.savefig(f"draw/{task}_example_prm_{i}.pdf", bbox_inches='tight', pad_inches=0.1,format="pdf")

print(pearsonr(evaluate_all_probs, evaluate_all_gts))