# python agentboard/eval_main.py --cfg-path eval_configs/alf-world/act_alfworld_gpt35.yaml --tasks alfworld --model gpt-35-turbo --log_path results/9_20_gpt35_act_alfworld
# python agentboard/eval_main.py --cfg-path eval_configs/alf-world/react_alfworld_gpt35.yaml --tasks alfworld --model gpt-35-turbo --log_path results/9_20_gpt35_react_alfworld


# python agentboard/eval_main.py --cfg-path eval_configs/pddl/act_pddl_gpt35.yaml --tasks pddl --model gpt-35-turbo --log_path results/9_20_gpt35_act_pddl


# python agentboard/eval_main.py --cfg-path eval_configs/alf-world/mpc_sample_alfworld_gpt35.yaml --tasks alfworld --model gpt-35-turbo --log_path results/9_21_gpt35_mpc_alfworld --max_num_steps 20

# python agentboard/eval_main.py --cfg-path eval_configs/alf-world/act_alfworld_gpt35.yaml --tasks alfworld --model llama-405b --log_path results/9_22_llama_405b_alfworld_act --max_num_steps 20

python agentboard/eval_main.py --cfg-path eval_configs/alf-world/react_alfworld_gpt35.yaml --tasks alfworld --model llama-405b --log_path results/9_22_llama_405b_alfworld_react --max_num_steps 20

python agentboard/eval_main.py --cfg-path eval_configs/pddl/act_pddl_gpt35.yaml --tasks pddl --model llama-405b --log_path results/9_22_llama_405b_pddl_act --max_num_steps 20

python agentboard/eval_main.py --cfg-path eval_configs/pddl/react_pddl_gpt35.yaml --tasks pddl --model llama-405b --log_path results/9_22_llama_405b_pddl_react --max_num_steps 20

