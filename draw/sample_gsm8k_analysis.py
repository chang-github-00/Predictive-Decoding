import numpy as np
import json
import os

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

file_autoregressive = f"results/run_anlaysis_important_results/{task}_autoregressive_output.jsonl"
autoregressive_results = load_analyze_data(task, file_autoregressive)
file_mpc = f"results/run_anlaysis_important_results/{task}_mpc_output.jsonl"
mpc_results = load_analyze_data(task, file_mpc)


correct_autoregressive = np.array([data["success_rate"] for data in autoregressive_results])
correct_mpc = np.array([data["success_rate"] for data in mpc_results])

corrected_samples = correct_mpc & np.logical_not(correct_autoregressive)

output_dir = "results/run_anlaysis_important_results/gsm8k_prm"



for i,item in enumerate(autoregressive_results):
    print(corrected_samples[i])
    if corrected_samples[i]:
        output = item["output"]
        answer_prefix = item["answer_prefix"]
        output_lines = output.split("\n")
        with open(f"{output_dir}corrected_{i}.txt", "w") as f:
            for line in output_lines:
                if line.startswith(answer_prefix):
                    f.write(f"{line}\t [+]\n")
                else:
                    f.write(f"{line}\t [ ]\n")