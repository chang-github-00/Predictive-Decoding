python agentboard/eval_main.py --cfg-path eval_configs/alf-world/act_alfworld_gpt35.yaml --tasks alfworld --model gpt-35-turbo --log_path results/9_20_gpt35_act_alfworld
python agentboard/eval_main.py --cfg-path eval_configs/alf-world/react_alfworld_gpt35.yaml --tasks alfworld --model gpt-35-turbo --log_path results/9_20_gpt35_react_alfworld


python agentboard/eval_main.py --cfg-path eval_configs/pddl/act_pddl_gpt35.yaml --tasks pddl --model gpt-35-turbo --log_path results/9_20_gpt35_act_pddl


python agentboard/eval_main.py --cfg-path eval_configs/alf-world/mpc_sample_alfworld_gpt35.yaml --tasks alfworld --model gpt-35-turbo --log_path results/9_21_gpt35_mpc_alfworld --max_num_steps 20