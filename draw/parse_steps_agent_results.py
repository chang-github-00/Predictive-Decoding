import re

# Sample data as a list of lines
file = '../results/9_20_gpt35_act_pddl/pddl.txt'
file = '../results/gpt35_react/pddl.txt'
file = '../results/gpt35_4_20_low_temperature/alfworld.txt'
file = '../results/9_22_llama_70b_pddl_mpc_sample/pddl.txt'
file = '../results/9_22_llama_70b_alfworld_mpc_sample/alfworld.txt'

data = []
max_steps = 20

with open(file, 'r') as f:
    for line in f:
        data.append(line.strip())
# Regular expression to capture success_rate and score_state
pattern = r'\[success_rate\]: (True|False).*?\[score_state\]: (\[.*?\])'

# Process each line
all_success_rates = []
all_progress_rates = []
for line in data:
    match = re.search(pattern, line)
    if match:
        success_rate = match.group(1)
        score_state_raw = match.group(2)

        # Convert score_state string to list of tuples
        score_state = eval(score_state_raw)

        last_progress_rate = 0
        for (step, reward) in score_state:
            if step > max_steps:
                break
            last_progress_rate = reward
        all_success_rates.append((last_progress_rate==1))
        all_progress_rates.append(last_progress_rate)
        
average_success_rate = sum(all_success_rates) / len(all_success_rates)
average_progress_rate = sum(all_progress_rates) / len(all_progress_rates)
print(f"Average Success Rate: {average_success_rate}")
print(f"Average Progress Rate: {average_progress_rate}")