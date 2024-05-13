import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

results_type = "dp_data_no_scratchpad_n_6_minval_-5_maxval_5_sampled_1000"

best_of_k_prefix = "../results/planning_gpt35_lookahead_best_of_k_"
lookahead_prefix = "../results/planning_gpt35_lookahead_light_"
lookahead_no_cache_prefix = "../results/planning_gpt35_lookahead_no_cache_"
tot_prefix = "../results/planning_gpt35_tot_"


# 生成一组数据
# n = 6
best_of_k = {2:0.33, 5:0.545, 10: 0.71, 20: 0.785}
lookahead = {1:0.429, 2:0.677, 5:0.835, 10: 0.891, 20: 0.918}
lookahead_no_cache = {2:0.25, 5:0.435, 10: 0.63, 20: 0.725}
tot = {3.33: 0.39, 10.67: 0.585}
mcts = {3.33: 0.62, 10.67: 0.75}

# 绘制原始数据和平滑曲线
plt.figure(figsize=(6, 5))

# plt.plot(list(lookahead.keys()), list(lookahead.values()), label='MPC w. n-gram', marker='o', color='red')
# plt.plot(list(lookahead_no_cache.keys()), list(lookahead_no_cache.values()), label='MPC wo. n-gram', marker='o', color='salmon')
# plt.plot(list(best_of_k.keys()), list(best_of_k.values()), label='Best of K', marker='o', color='steelblue')
# plt.plot(list(tot.keys()), list(tot.values()), label='TOT', marker='o',color='darkcyan')
# plt.plot(list(mcts.keys()), list(mcts.values()), label='MCTS', marker='o',color='royalblue')
colors = sns.color_palette("coolwarm",5)
colors = sns.color_palette("deep", 5)
sns.set(style="ticks")
sns.lineplot(x=list(lookahead.keys()), y=list(lookahead.values()), label='MPC w. n-gram', marker='o', color=colors[0], linewidth=3, markersize=10)
sns.lineplot(x=list(lookahead_no_cache.keys()), y=list(lookahead_no_cache.values()), label='MPC wo. n-gram', marker='o', color=colors[-1], linewidth=3, markersize=10)
sns.lineplot(x=list(best_of_k.keys()), y=list(best_of_k.values()), label='Self-Consistency', marker='*', color=colors[-3], linewidth=3, markersize=15)
sns.lineplot(x=list(mcts.keys()), y=list(mcts.values()), label='MCTS', marker='s', color=colors[-2], linewidth=3, markersize=10)
sns.lineplot(x=list(tot.keys()), y=list(tot.values()), label='TOT', marker='d', color=colors[1], linewidth=3, markersize=12)

# draw horizontal line
plt.axhline(y=0.36, color='black', linestyle='--', label='COT', linewidth=3)
plt.xlabel("Sampling Parameter K", fontsize=15)
plt.ylabel("Accuracy", fontsize=15)
plt.xticks(range(2, 21, 4), fontsize=12)
plt.yticks([0.1, 0.3, 0.5, 0.7, 0.9], fontsize=12)
plt.legend(prop={'size': 14})
plt.savefig(f"dp_6.pdf")


dp_rollout_length = {0:0.35, 1:0.597, 2:0.709, 3:0.77, 4:0.85, 5:0.91}
pf_rollout_length = {0:0.31, 1:0.55, 2:0.67, 3:0.73, 4:0.81, 5:0.87} 
plt.figure(figsize=(6, 5)) 
colors = [ "#456990", "#d87659"]
data = {'Rollout Length': list(dp_rollout_length.keys()) + list(pf_rollout_length.keys()), 'Accuracy': list(dp_rollout_length.values()) + list(pf_rollout_length.values()), 'Task': ['Dynamic Programming'] * 6 + ['Path Finding'] * 6}
sns.barplot(x='Rollout Length', y='Accuracy', hue='Task', data=data, palette=colors, linewidth=2, edgecolor=".3")
# sns.lineplot(x=list(dp_rollout_length.keys()), y=list(dp_rollout_length.values()), label='MPC, DP   (N=6)', marker='o', color=colors[0], linewidth=2)
# sns.lineplot(x=list(pf_rollout_length.keys()), y=list(pf_rollout_length.values()), label='MPC, Path (N=6)', marker='o', color=colors[1], linewidth=2)
plt.xlabel("MPC Rollout Length", fontsize=15)
plt.ylabel("Accuracy", fontsize=15)
plt.xticks(range(6), fontsize=12)
plt.yticks([0.1, 0.3, 0.5, 0.7, 0.9], fontsize=12)
plt.legend(prop={'size': 14})
plt.savefig(f"rollout_length.pdf")