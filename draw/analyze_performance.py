import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np


predictive_decoding_1 = {
    't': [1,2,4,6],
    "flops": [27, 54, 108, 161],
    "Acc": [77.0, 76.4, 77.9, 78.5]
}

autoregressive = {"flops": 7.5, "Acc": 71.3}

beam_search = {'n': [ 1, 2, 4, 8], "flops": [7.5, 30, 120, 480],"Acc": [71.3, 76.5,77.9, 78.4 ]}

guided_decoding = {"flops": 119, "Acc": 63.9}


sns.set_style("whitegrid", {"grid.linestyle": "--", "grid.color": "0.7", "axes.edgecolor": "0"})

plt.figure(figsize=(6, 4))
size = [50, 75, 100,200]
plt.scatter(predictive_decoding_1['flops'], predictive_decoding_1['Acc'], color='#3fecd3',marker='^', s=size, label="Predictive Decoding",edgecolors='black')
# fit a curve to the data points
for i in range(len(predictive_decoding_1['t'])):
    plt.text(predictive_decoding_1['flops'][i]+5*2**i, predictive_decoding_1['Acc'][i], f"T₀={predictive_decoding_1['t'][i]}", fontsize=10, color='black')

size = [50, 75,100, 200]
plt.scatter(beam_search['flops'], beam_search['Acc'], color='orange', marker='*', label="Beam Search",s=size,edgecolors='black')
z = np.polyfit(beam_search['flops'], beam_search['Acc'], 3)
p = np.poly1d(z)
plt.plot(beam_search['flops'],p(beam_search['flops']), color='orange', linestyle='--')



plt.xlabel("Inference FLOPS (10^12)")
plt.ylabel("Accuracy (%)")

plt.title("Performance v.s. Efficiency")
plt.xscale('log')
plt.xticks([10, 20, 40, 80, 160, 320, 640], [10, 20, 40, 80, 160, 320, 640])

plt.scatter(autoregressive['flops'], autoregressive['Acc'], color='green', marker='p', s=75, label="Autoregressive",edgecolors='black')


# plt.scatter(guided_decoding['flops'], guided_decoding['Acc'], color='orange', marker='p', s=100, label="Guided Decoding")

plt.legend(loc='lower right')
plt.savefig("performance_vs_flops.pdf", bbox_inches='tight', format='pdf')