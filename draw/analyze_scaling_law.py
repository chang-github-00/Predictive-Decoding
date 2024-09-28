import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import random

def update_mean_std(data):
    mean = []
    std = []
    for i in range(len(data)):
        if type(data[i]) == list:
            mean.append(np.mean(data[i]))
            std_ = np.std(data[i])/2
            std_  = min(std_, 0.5)
            std.append(std_) # std/2
        else:
            mean.append(data[i])
            std.append(0)
            
    return mean, std
    

rank = {
    "flops": [312, 162, 81, 44, 21, 10.1, 5.6],
    "all_performance": [[85.2,86.2,86.2], [85.1, 85.7,87.3], [84.5,85.7,87.3], [86.2, 85.7, 84.6], [80.4, 80.4, 84.1], [78.3,80.1,78.3], [75.1,75.7,76.7]]
}

vote = {
    "flops": [312, 162, 81, 44, 21, 10.1, 5.6],
    "all_performance": [[87.7, 86.2], [87.3, 85.7, 85.7], [85.7, 86.2, 85.7], [82.8, 85.2, 84.1], [82.0, 83.1, 84.7], [77.2, 77.2, 77.7], [75.1,75.7,76.7]]
}

mcts = {
    "flops": [435, 294, 172],
    "all_performance": [85.7, 83.0, 80.4]
}

predictive_decoding = {
    "flops": [360, 177, 86, 44],
    "all_performance": [[88.8, 89.9,90.0], [87.8, 88.3, 87.8], [87.3,87.3,88.4], [87.3, 84.7, 87.3, 83.6]]
}

predictive_decoding_t = {
    "flops": [11, 22, 29, 44],
    "all_performance": [[78.3,80.1,79.9],[83.6,84.7,83.6],[84.6, 82.0, 88.4],[87.3, 84.7, 87.3, 83.6]]
}
guided_decoding = {
    "flops": [383, 276, 163, 108],
    "all_performance": [87.8, 86.7, 84.9,81.1]
}

sns.set_style("whitegrid", {"grid.linestyle": "--", "grid.color": "0.7", "axes.edgecolor": "0"})

plt.figure(figsize=(6, 4))


# lineplot
plt.xscale('log')


# calculate the mean and std
rank['performance'], rank['std'] = update_mean_std(rank['all_performance'])

mcts['performance'], mcts['std'] = update_mean_std(mcts['all_performance'])
predictive_decoding['performance'], predictive_decoding['std'] = update_mean_std(predictive_decoding['all_performance'])
guided_decoding['performance'], guided_decoding['std'] = update_mean_std(guided_decoding['all_performance'])

vote['performance'], vote['std'] = update_mean_std(vote['all_performance'])

predictive_decoding_t['performance'], predictive_decoding_t['std'] = update_mean_std(predictive_decoding_t['all_performance'])




plt.plot(rank['flops'], rank['performance'], color='#ffc529',marker='o', label="Autorgressive + Rank",markersize=4)
plt.fill_between(rank['flops'], np.array(rank['performance']) - np.array(rank['std']), np.array(rank['performance']) + np.array(rank['std']), color='#ffc529', alpha=0.2)

plt.plot(vote['flops'], vote['performance'], color='#069099', marker='o', label="Autorgressive + Weighted Voting",markersize=4)
plt.fill_between(vote['flops'], np.array(vote['performance']) - np.array(vote['std']), np.array(vote['performance']) + np.array(vote['std']), color='#069099', alpha=0.2)


plt.plot(guided_decoding['flops'], guided_decoding['performance'], color='#f6630c', marker='o', label="Guided Decoding",markersize=4)
plt.fill_between(guided_decoding['flops'], np.array(guided_decoding['performance']) - np.array(guided_decoding['std']), np.array(guided_decoding['performance']) + np.array(guided_decoding['std']), color='#f6630c', alpha=0.2)


plt.plot(mcts['flops'], mcts['performance'], color='#7bcba2', marker='o', label="MCTS",markersize=4)
plt.fill_between(mcts['flops'], np.array(mcts['performance']) - np.array(mcts['std']), np.array(mcts['performance']) + np.array(mcts['std']), color='#7bcba2', alpha=0.2)

plt.plot(predictive_decoding['flops'], predictive_decoding['performance'], color='#cf0f5b', marker='o', label="Predictive Decoding (T₀=6, K=2,4,8,16)",markersize=4)
plt.fill_between(predictive_decoding['flops'], np.array(predictive_decoding['performance']) - np.array(predictive_decoding['std']), np.array(predictive_decoding['performance']) + np.array(predictive_decoding['std']), color='#cf0f5b', alpha=0.2)


plt.plot(predictive_decoding_t['flops'], predictive_decoding_t['performance'], color='#cf0f5b', marker='o', label="Predictive Decoding (K=2,T₀=1,2,4,6)",markersize=4, linestyle='--')
plt.fill_between(predictive_decoding_t['flops'], np.array(predictive_decoding_t['performance']) - np.array(predictive_decoding_t['std']), np.array(predictive_decoding_t['performance']) + np.array(predictive_decoding_t['std']), color='#cf0f5b', alpha=0.2)


plt.legend( loc='lower right', fontsize=8)

plt.xticks([4, 8, 16, 32, 64, 128, 256, 512], [4, 8, 16, 32, 64, 128, 256, 512])


plt.xlabel("Inference FLOPS (10^12)") 
plt.ylabel("Accuracy (%)")
plt.title("Inference Scaling Law on GSM8K")   

plt.savefig("scaling_law.pdf", bbox_inches='tight', format='pdf')