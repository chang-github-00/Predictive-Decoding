import matplotlib.pyplot as plt
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

task = "math"#"gsm8k"
prob_data_path = f"results/run_anlaysis_important_results/{task}_output_probs.jsonl"
with open(prob_data_path, "r") as f:
    prob = json.load(f)

file_autoregressive = f"results/run_anlaysis_important_results/{task}_autoregressive_output.jsonl"
autoregressive_results = load_analyze_data(task, file_autoregressive)
file_mpc = f"results/run_anlaysis_important_results/{task}_mpc_output.jsonl"
mpc_results = load_analyze_data(task, file_mpc)

autoregressive_data = np.array([p[0] for p in prob])
beamsearch_data = np.array([p[1] for p in prob])
mpc_data = np.array([p[2] for p in prob])

diff_data = mpc_data - autoregressive_data
diff_data_2 = beamsearch_data - autoregressive_data
correct = np.array([bool(item["success_rate"]) for item in autoregressive_results])
wrong = np.logical_not(correct)
activated = ((diff_data >0) + (diff_data_2 > 0))>0
non_activated = np.logical_not(activated)

# calculate confusion matrix
TP = np.sum(activated & wrong)
FP = np.sum(activated & correct)
TN = np.sum(non_activated & wrong)
FN = np.sum(non_activated & correct)
print(f"TP: {TP}, FP: {FP}, TN: {TN}, FN: {FN}")

activated_2 = autoregressive_data < 0.95
non_activated_2 = np.logical_not(activated_2)
# calculate confusion matrix
TP = np.sum(activated_2 & wrong)
FP = np.sum(activated_2 & correct)
TN = np.sum(non_activated_2 & wrong)
FN = np.sum(non_activated_2 & correct)
print(f"TP: {TP}, FP: {FP}, TN: {TN}, FN: {FN}")

# draw the histogram of the output probabilities
plt.figure(figsize=(10, 6))
plt.hist(autoregressive_data, bins=100, alpha=0.5, label='autoregressive')
# plt.hist(beamsearch_data, bins=100, alpha=0.5, label='beamsearch')
plt.hist(mpc_data, bins=100, alpha=0.5, label='mpc')
plt.xlabel('output probability')
plt.ylabel('frequency')
plt.legend(loc='upper right')
plt.savefig("draw/output_probs.png")

print(autoregressive_data.mean())
print(beamsearch_data.mean())
print(mpc_data.mean())

# draw the histogram of the difference between mpc and autoregressive
plt.figure(figsize=(10, 6))
plt.hist(diff_data, bins=100, alpha=0.5, label='mpc - autoregressive')
plt.xlabel('difference')
plt.ylabel('frequency')
plt.legend(loc='upper right')
plt.savefig("draw/diff_probs.png")

# draw the historgram of difference between correct and wrong
plt.figure(figsize=(10, 6))
plt.hist(autoregressive_data[correct], bins=100, alpha=0.5, label='correct')
plt.hist(autoregressive_data[wrong], bins=100, alpha=0.5, label='wrong')
plt.xlabel('output probability')
plt.ylabel('frequency')
plt.legend(loc='upper right')
plt.savefig("draw/output_probs_correct_wrong.png")

print(autoregressive_data[correct].mean())
print(autoregressive_data[wrong].mean())


correct_mpc = np.array([bool(item["success_rate"]) for item in mpc_results])
wrong_mpc = np.logical_not(correct_mpc)
plt.figure(figsize=(10, 6))
plt.hist(mpc_data[correct], bins=100, alpha=0.5, label='correct')
plt.hist(mpc_data[wrong], bins=100, alpha=0.5, label='wrong')
plt.xlabel('output probability')
plt.ylabel('frequency')
plt.legend(loc='upper right')
plt.savefig("draw/output_probs_correct_wrong_mpc.png")

# draw the histogram of difference between TP and FP
plt.figure(figsize=(10, 6))
plt.hist(diff_data[correct_mpc & wrong], bins=100, alpha=0.5, label='FP')
print((diff_data[correct_mpc & wrong]>0).sum()/len(diff_data[correct_mpc & wrong]))
# plt.hist(diff_data[activated & correct], bins=100, alpha=0.5, label='TP')

plt.legend(loc='upper right')
plt.xlabel('difference')
plt.ylabel('frequency')
plt.savefig("draw/diff_probs_TP_FP.png")