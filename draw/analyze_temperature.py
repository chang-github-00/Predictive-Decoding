import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np



data = {
    'alpha': [0.6, 0.6, 0.6, 0.6, 0.6, 0.6],
    'tau': [0.01, 0.05, 0.1, 0.5, 1.0, 5.0],
    'selfbleu': [0.979, 0.969, 0.975, 0.954, 0.952, 0.951],
    'rouge': [0.977, 0.963, 0.954, 0.940, 0.939, 0.938],
    'pass@1': [56.8, 53.8, 48.9, 48.2, 48.8, 49.1],
    'std': [0.8, 2.2, 0.4, 0.4, 0.4, 0.9]
}

data["rouge"] = [1-x for x in data["rouge"]]

data["std"] = [x / 1.5 for x in data["std"]]

data_2 = {
    'alpha': [0.1, 0.3, 0.6, 0.8, 1.0],
    'tau': [0.05] * 5,
    'selfbleu': [0.988, 0.976, 0.969, 0.952, 0.954],
    'rouge': [0.990, 0.975, 0.963, 0.944, 0.940],
    'pass@1': [54.0, 56.1, 53.8, 48.8, 44.1],
    'std': [0.47, 1.32, 2.2, 0.25, 0.3]
}
data_2["rouge"] = [1-x for x in data_2["rouge"]]
data_2["std"] = [x / 1.5 for x in data_2["std"]]
    
# Create a DataFrame
df = pd.DataFrame(data)
df_2 = pd.DataFrame(data_2)



print(df)

# draw lineplot based on rouge and pass@1 also include std



df = df.sort_values(by='rouge')
df_2 = df_2.sort_values(by='rouge')


# sns.set(style="whitegrid")
# grid line dotted
sns.set_style("whitegrid", {"grid.linestyle": "--", "grid.color": "0.7", "axes.edgecolor": "0"})

plt.figure(figsize=(6, 4))

# scatter all points in the first plot
plt.scatter(df['rouge'], df['pass@1'], color='blue',marker='^')

plt.scatter(df_2['rouge'], df_2['pass@1'], color='red')

# add dot tau \infty, alpha = 0.6, diversity = 0.23, pass@1 = 44.3
plt.scatter(0.023, 44.1, color='green', marker='p', s=100)
plt.text(0.023, 44.4, f"Autoregressive Decoding, α=0.6", fontsize=10, color='black')
plt.scatter(0.02, 53.5, color='orange', marker='*', s=100)
plt.text(0.02, 54.0, f"Beam Search", fontsize=10, color='black')

# write alpha and tau values for each point
for i in range(len(df)):
    if i == 4: continue
    # move the text a little bit to the right
    plt.text(df['rouge'][i] + 0.001, df['pass@1'][i], f"τ={df['tau'][i]}", fontsize=10, color='black')

for i in range(len(df_2)):  
    if i in [2,3]: continue
    plt.text(df_2['rouge'][i] + 0.001, df_2['pass@1'][i] -0.3, f"α={df_2['alpha'][i]}", fontsize=10, color='black')

plt.text(0.055, 49, f"α=0.8", fontsize=10, color='black')
# plt.text(0.02, 53.5, f"α=0.8", fontsize=10, color='black')

# fit a curve to the data points
z = np.polyfit(df['rouge'], df['pass@1'], 3)
p = np.poly1d(z)
plt.plot(df['rouge'], p(df['rouge']), "b--", label='Predictive Decoding α=0.6')


z = np.polyfit(df_2['rouge'], df_2['pass@1'], 3)
p = np.poly1d(z)
plt.plot(df_2['rouge'], p(df_2['rouge']), "r--",label='Predictive Decoding τ=0.05')


# plt axis line opacity
plt.gca().spines['top'].set_alpha(1.0)
plt.gca().spines['right'].set_alpha(1.0)
plt.gca().spines['left'].set_alpha(1.0)
plt.gca().spines['bottom'].set_alpha(1.0)
plt.legend()

plt.ylabel('Pass@1')
plt.xlabel('Diversity')
plt.title('Performance v.s. Diversity')
plt.savefig('pass@1_vs_alpha.pdf', format='pdf', bbox_inches='tight')