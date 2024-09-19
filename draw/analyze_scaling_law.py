import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np



rank = {
    "flops": [312, 162, 81, 44, 21, 5.6],
    "performance": [86.2, 87.3, 84.5, 86.2, 80.4, 75.6]
}

mcts = {
    "flops": [435, 294, 172],
    "performance": [83.1, 80.4, 78.3]
}

predictive_decoding = {
    "flops": [360, 177, 86, 44],
    "performance": [89.9, 87.8, 87.3, 83.6]
}

guided_decoding = {
    "flops": [383, 276, 108],
    "performance": [87.8, 86.7, 81.1]
}

sns.set_style("whitegrid", {"grid.linestyle": "--", "grid.color": "0.7", "axes.edgecolor": "0"})

plt.figure(figsize=(6, 4))


# lineplot
plt.xscale('log')


plt.plot(predictive_decoding['flops'], predictive_decoding['performance'], color='#cf0f5b', marker='D', label="Predictive Decoding")

plt.plot(rank['flops'], rank['performance'], color='#069099',marker='o', label="Autorgressive + Rank")

plt.plot(guided_decoding['flops'], guided_decoding['performance'], color='#f6630c', marker='p', label="Guided Decoding")


plt.plot(mcts['flops'], mcts['performance'], color='#7bcba2', marker='s', label="MCTS")


plt.legend()

plt.xticks([4, 8, 16, 32, 64, 128, 256, 512], [4, 8, 16, 32, 64, 128, 256, 512])


plt.xlabel("Inference FLOPS (10^12)") 
plt.ylabel("Accuracy (%)")
plt.title("Inference Scaling Law on GSM8K")   

plt.savefig("scaling_law.pdf", bbox_inches='tight', format='pdf')