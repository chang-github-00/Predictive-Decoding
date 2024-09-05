import matplotlib.pyplot as plt
import numpy as np
import json
import os
import seaborn as sns

def load_analyze_data(task, path='result/...'):
    examples = []
    path = os.path.join(path)
    with open(path, "r") as f:
        for line in f:
            js = json.loads(line)
            examples.append(js)
    if task == "humaneval": examples = examples[:164]
    return examples

task = "math"
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

optimal_data = np.max([autoregressive_data, beamsearch_data, mpc_data], axis=0)

# Create a figure and axis
plt.figure(figsize=(10, 5))

# Plot density plots for each array
# plt.hist(beamsearch_data[beamsearch_data>0.05], bins=100, alpha=0.5, label='Beam Search',density=True)
plt.hist(optimal_data, bins=100, alpha=0.6, label='LLM Non-myopic Planning',density=True)
plt.hist(autoregressive_data, bins=100, alpha=0.5, label='LLM Autoregressive Planning',density=True)

plt.legend(loc='upper right', fontsize=18)

# Calculate and plot medians
autoregressive_median = np.median(autoregressive_data)
optimal_median = np.median(optimal_data)

plt.axvline(autoregressive_median, color='orange', linestyle='-', label='Autoregressive Median',linewidth=3)
# plt.axvline(beamsearch_median, color='orange', linestyle='--', label='Beam Search Median')
plt.axvline(optimal_median,  linestyle='-', color='cornflowerblue',label='MPC Median',linewidth=3)
plt.text(autoregressive_median-0.01, 3.5, f'{autoregressive_median:.2f}',fontdict={'size': 15})
plt.text(optimal_median, 3.5, f'{optimal_median:.2f}', fontdict={'size': 15})
plt.ylabel('Density', fontsize=18)
plt.xlabel('Generation Probability P (MATH)', fontsize=18)
plt.xlim(0, 1)
plt.savefig(f"draw/{task}_output_probs.pdf", bbox_inches='tight', pad_inches=0.1,format="pdf")


# create a figure to plot difference
diff_data = optimal_data - autoregressive_data

correct = np.array([data["success_rate"] for data in autoregressive_results])
wrong = np.logical_not(correct)

plt.figure(figsize=(5, 5))


sns.kdeplot(diff_data[wrong], label='Wrong', color='red', linewidth=3)
sns.kdeplot(diff_data[correct], label='Correct', color='blue', linewidth=3)
# sns.distplot(diff_data, label='Overall', kde=True, bins=50,  kde_kws={'linewidth': 2})


plt.xlabel('Myopic Gap p* (MATH)', fontsize=18)
plt.ylabel('Density', fontsize=18)
plt.legend(loc='upper right', fontsize=18)
plt.xticks(fontsize=12)
plt.yticks(fontsize=12)
plt.xlim(-0.001, 0.3)
plt.savefig(f"draw/{task}_output_probs_diff.pdf", bbox_inches='tight', pad_inches=0.1,format="pdf")


activated = diff_data > 0.01
non_activated = np.logical_not(activated)

# calculate confusion matrix
TP = np.sum(activated & wrong)
FP = np.sum(activated & correct)
TN = np.sum(non_activated & wrong)
FN = np.sum(non_activated & correct)
print(f"TP: {TP}, FP: {FP}, TN: {TN}, FN: {FN}")
print(f"{TP/(TP+TN)}")
print(f"{FP/(FP+FN)}")







task = "gsm8k"
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

optimal_data = np.max([autoregressive_data, beamsearch_data, mpc_data], axis=0)

# Create a figure and axis
plt.figure(figsize=(10, 5))

# Plot density plots for each array
# plt.hist(beamsearch_data[beamsearch_data>0.05], bins=100, alpha=0.5, label='Beam Search',density=True)
plt.hist(optimal_data, bins=200, alpha=0.6, label='LLM Non-myopic Planning',density=True)
plt.hist(autoregressive_data, bins=200, alpha=0.5, label='LLM Autoregressive Planning',density=True)

plt.legend(loc='upper left', fontsize=18)

# Calculate and plot medians
autoregressive_median = np.median(autoregressive_data)
optimal_median = np.median(optimal_data)

plt.axvline(autoregressive_median, color='orange', linestyle='-', label='Autoregressive Median',linewidth=3)
# plt.axvline(beamsearch_median, color='orange', linestyle='--', label='Beam Search Median')
plt.axvline(optimal_median,  linestyle='-', color='cornflowerblue',label='MPC Median',linewidth=3)
plt.text(autoregressive_median-0.025, 18, f'{autoregressive_median:.2f}',fontdict={'size': 15})
plt.text(optimal_median, 18, f'{optimal_median:.2f}', fontdict={'size': 15})
plt.ylabel('Density', fontsize=18)
plt.xlabel('Generation Probability P (GSM8K)', fontsize=18)
plt.xlim(0.7, 1)
plt.savefig(f"draw/{task}_output_probs.pdf", bbox_inches='tight', pad_inches=0.1,format="pdf")



# create a figure to plot difference
diff_data = optimal_data - autoregressive_data

correct = np.array([data["success_rate"] for data in autoregressive_results])
wrong = np.logical_not(correct)

plt.figure(figsize=(5, 5))


sns.kdeplot(diff_data[wrong], label='Wrong', color='red', linewidth=3)
sns.kdeplot(diff_data[correct], label='Correct', color='blue', linewidth=3)
# sns.distplot(diff_data, label='Overall', kde=True, bins=50,  kde_kws={'linewidth': 2})

plt.xlim(-0.001, 0.4)
plt.xlabel('Myopic Gap p* (GSM8K)', fontsize=18)
plt.ylabel('Density', fontsize=18)
plt.legend(loc='upper right', fontsize=18)
plt.xticks(fontsize=12)
plt.yticks(fontsize=12)
plt.savefig(f"draw/{task}_output_probs_diff.pdf", bbox_inches='tight', pad_inches=0.1,format="pdf")


activated = diff_data > 0.01
non_activated = np.logical_not(activated)

# calculate confusion matrix
TP = np.sum(activated & wrong)
FP = np.sum(activated & correct)
TN = np.sum(non_activated & wrong)
FN = np.sum(non_activated & correct)
print(f"TP: {TP}, FP: {FP}, TN: {TN}, FN: {FN}")
print(f"{TP/(TP+TN)}")
print(f"{FP/(FP+FN)}")



