import matplotlib.pyplot as plt
import numpy as np
import json
import os
import seaborn as sns
import matplotlib.cm as cm
from scipy.stats import pearsonr, gaussian_kde
from matplotlib.colors import Normalize


def calculate_ece(probs, labels, num_bins=5):
    """
    Calculate the Expected Calibration Error (ECE).

    :param probs: numpy array of predicted probabilities
    :param labels: numpy array of true labels (0 or 1)
    :param num_bins: Number of bins to use for calibration
    :return: ECE value
    """
    labels = labels > 0.5
    bins = np.linspace(0, 1, num_bins + 1)
    bin_indices = np.digitize(probs, bins) - 1

    ece = 0.0
    for i in range(num_bins):
        # Indices of samples in the current bin
        bin_mask = bin_indices == i
        bin_size = np.sum(bin_mask)

        if bin_size > 0:
            # Average predicted probability in the bin
            avg_prob = np.mean(probs[bin_mask])

            # Average true outcome in the bin
            avg_true = np.mean(labels[bin_mask])

            # Contribution of this bin to ECE
            ece += (bin_size / len(probs)) * np.abs(avg_prob - avg_true)

    return ece



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

first_false_probs = []

# calculate the average prob of all full correct examples
all_correct_probs = []
for i,prob in enumerate(probs):
    all_probs = [p[0] for p in prob]
    all_gts = [p[1] for p in prob]
    
    all_probs = [p for p in all_probs if np.isnan(p) == False]
    all_gts = all_gts[:len(all_probs)]
    evaluate_all_probs.extend(all_probs)
    evaluate_all_gts.extend(all_gts)
    
    is_all_correct = False not in all_gts
    
    if len(all_probs) == 0:
        continue
    
    if is_all_correct:
        all_correct_probs.extend(all_probs)

all_correct_probs = np.array(all_correct_probs)

# calculate the frequency of all correct examples in each bin
# make the bins smoother with kdeplot
correct_value_freq, correct_value= np.histogram(all_correct_probs, bins=20)
correct_value_freq = correct_value_freq / len(all_correct_probs)

# smooth correct_value_freq
kde = gaussian_kde(all_correct_probs)
evaluated = kde.evaluate(np.linspace(0, 1, 100))

# plot evaluation
plt.figure(figsize=(4, 2))
plt.plot(np.linspace(0, 1, 100), evaluated, label='All Correct Examples', marker='o', markersize=3)
plt.xlabel('LLM Score')
plt.ylabel('Density')
plt.savefig(f"draw/{task}_corrected_all_correct_prm.pdf", bbox_inches='tight', pad_inches=0.1,format="pdf")


# generate the value of continuos heatmap for all correct examples
# kde_probs = sns.kdeplot(all_correct_probs, shade=True, color='blue', label='All Correct Examples', alpha=0.5)
# print(correct_value_freq)

for i,prob in enumerate(probs):
    all_probs = [p[0] for p in prob]
    all_gts = [p[1] for p in prob]
    
    all_probs = [p for p in all_probs if np.isnan(p) == False]
    all_gts = all_gts[:len(all_probs)]
    evaluate_all_probs.extend(all_probs)
    evaluate_all_gts.extend(all_gts)
    
    is_all_correct = False not in all_gts
    if is_all_correct:
        continue
    
    first_false = max(all_gts.index(False)-2, 0)
    
    
    max_steps = min(8, len(all_probs)-first_false)
    
    false_prob= all_probs[first_false:first_false+max_steps]
    
    if max_steps < 9:
        for i in range(9-max_steps):
            false_prob.append(all_probs[-1])
    false_prob = np.array(false_prob)

    
    
    first_false_probs.append(false_prob)

first_false_probs = np.array(first_false_probs)

average_first_false_probs = np.mean(first_false_probs, axis=0)
norm = Normalize(vmin=evaluated.min(), vmax=evaluated.max())
cmap = cm.get_cmap('Greens')
plt.figure(figsize=(4,3))
plt.plot(average_first_false_probs, label='Incorrect Answer LogP w.r.t steps', marker='o', markersize=3)
plt.axvline(1, color='red', linestyle='--', label='First Incorrect Step')
evaluate_range = np.linspace(0, 1, 100)

for i in range(98):
    plt.fill_between(range(9), evaluate_range[i], evaluate_range[i+1], color=cmap(norm(evaluated[i])), alpha=0.5, edgecolor=None)

plt.fill_between(range(9), evaluate_range[98], evaluate_range[99], color=cmap(norm(evaluated[99])), alpha=0.5, edgecolor=None, label='Correct Steps LogP Density')


plt.xlabel('Step')
plt.ylabel('LLM Score')
plt.ylim(0.86, 0.98)



plt.legend(fontsize=8, loc='upper right')

plt.savefig(f"draw/{task}_average_first_false_prm.pdf", bbox_inches='tight', pad_inches=0.1,format="pdf")
    
final_probs = []
final_gts = []

for i,prob in enumerate(probs):
    all_probs = [p[0] for p in prob]
    all_gts = [p[1] for p in prob]
    
    all_probs = [p for p in all_probs if np.isnan(p) == False]
    all_gts = all_gts[:len(all_probs)]
    evaluate_all_probs.extend(all_probs)
    evaluate_all_gts.extend(all_gts)
    if len(all_probs) == 0:
        continue
    final_probs.append(all_probs[-1])
    final_gts.append(all_gts[-1])
    # if i not in [3,10,43,52,56]:
    #     continue
    # if 0 in all_gts:


    #     first_false = max(all_gts.index(False)-1, 0)
    #     plt.figure(figsize=(4, 2))
    #     plt.plot(all_probs, label='Probability of Correct Answer', marker='o', markersize=3)
    #     # draw a line to indicate the first false
    #     plt.axvline(first_false, color='red', linestyle='--', label='First Incorrect Answer')
    #     plt.xlabel('Step')
    #     plt.ylabel('LLM Score') 
    #     plt.savefig(f"draw/{task}_example_prm_{i}.pdf", bbox_inches='tight', pad_inches=0.1,format="pdf")
    # else:
    #     plt.figure(figsize=(4, 2))
    #     plt.plot(all_probs, label='Probability of Correct Answer', marker='o', markersize=3)
    #     plt.xlabel('Step')
    #     plt.ylabel('LLM Score') 
    #     plt.savefig(f"draw/{task}_example_prm_{i}.pdf", bbox_inches='tight', pad_inches=0.1,format="pdf")

print(pearsonr(evaluate_all_probs, evaluate_all_gts))
print(pearsonr(final_probs, final_gts))
print(calculate_ece(np.array(evaluate_all_probs), np.array(evaluate_all_gts)))
print(calculate_ece(np.array(final_probs), np.array(final_gts)))

# Example usage
