# dynamic programming
# CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 python agentboard/eval_main.py --cfg-path eval_configs/react_results_all_tasks.yaml --model deepseek-67b --tasks alfworld pddl --log_path results/deepseek_react



# python agentboard/eval_planning.py --algorithm COT --cfg-path eval_configs/planning_results_all_tasks.yaml --model gpt-35-turbo --tasks dp --log_path results/planning_gpt35 --data_path data/dp/data_no_scratchpad_n_10_minval_-5_maxval_5_sampled_1000.jsonl --prompt_path agentboard/prompts/Planning/dp_cot.json

# python agentboard/eval_planning.py --cfg-path eval_configs/planning_results_all_tasks.yaml --algorithm TOT --model gpt-35-turbo --tasks dp --log_path results/planning_gpt35_tot --data_path data/dp/data_no_scratchpad_n_6_minval_-5_maxval_5_sampled_1000.jsonl --prompt_path agentboard/prompts/Planning/dp_tot.json 

# python agentboard/eval_planning.py --cfg-path eval_configs/planning_results_all_tasks.yaml --algorithm MCTS_Light --model gpt-35-turbo --tasks dp --log_path results/planning_gpt35_mcts_64_2 --data_path data/dp/data_no_scratchpad_n_10_minval_-5_maxval_5_sampled_1000.jsonl --prompt_path agentboard/prompts/Planning/dp_lookahead_light.json 

# python agentboard/eval_planning.py --cfg-path eval_configs/planning_results_all_tasks.yaml --algorithm TOT_Light --model gpt-35-turbo --tasks dp --log_path results/planning_gpt35_tot_64 --data_path data/dp/data_no_scratchpad_n_10_minval_-5_maxval_5_sampled_1000.jsonl --prompt_path agentboard/prompts/Planning/dp_tot.json 

# python agentboard/eval_planning.py --cfg-path eval_configs/planning_results_all_tasks.yaml --algorithm BestK --model gpt-35-turbo --tasks dp --log_path results/planning_gpt35_lookahead_best_of_k_10 --data_path data/dp/data_no_scratchpad_n_10_minval_-5_maxval_5_sampled_1000.jsonl --prompt_path agentboard/prompts/Planning/dp_vanilla.json

# python agentboard/eval_planning.py --cfg-path eval_configs/planning_results_all_tasks.yaml --algorithm Lookahead_Eval_Ablation --model gpt-35-turbo --tasks dp --log_path results/planning_gpt35_lookahead_no_cache_10 --data_path data/dp/data_no_scratchpad_n_10_minval_-5_maxval_5_sampled_1000.jsonl --prompt_path agentboard/prompts/Planning/dp_lookahead_light.json 


# python agentboard/eval_planning.py --cfg-path eval_configs/planning_results_all_tasks.yaml --algorithm Lookahead_Eval_Ablation --model gpt-35-turbo --tasks dp --log_path results/planning_gpt35_lookahead_light_10_2 --data_path data/dp/data_no_scratchpad_n_10_minval_-5_maxval_5_sampled_1000.jsonl --prompt_path agentboard/prompts/Planning/dp_lookahead_light.json 

# python agentboard/eval_planning.py --cfg-path eval_configs/planning_results_all_tasks.yaml --algorithm Lookahead_Eval_Local --model gpt-35-turbo --tasks dp --log_path results/planning_gpt35_lookahead_local_20_1 --data_path data/dp/data_no_scratchpad_n_6_minval_-5_maxval_5_sampled_1000.jsonl --prompt_path agentboard/prompts/Planning/dp_lookahead_light.json 

# python agentboard/eval_planning.py --cfg-path eval_configs/planning_results_all_tasks.yaml --algorithm Lookahead_Eval_Local --model gpt-35-turbo --tasks dp --log_path results/planning_gpt35_lookahead_local_20_0 --data_path data/dp/data_no_scratchpad_n_6_minval_-5_maxval_5_sampled_1000.jsonl --prompt_path agentboard/prompts/Planning/dp_lookahead_light.json --lookahead_length 0
# python agentboard/eval_planning.py --cfg-path eval_configs/dp/planning_results_all_tasks.yaml --algorithm Lookahead_Eval_Local --model gpt-35-turbo --tasks dp --log_path results/planning_gpt35_lookahead_local_20_4 --data_path data/dp/data_no_scratchpad_n_6_minval_-5_maxval_5_sampled_1000.jsonl --prompt_path agentboard/prompts/Planning/dp_lookahead_light.json --lookahead_length 4
# python agentboard/eval_planning.py --cfg-path eval_configs/dp/planning_results_all_tasks.yaml --algorithm Lookahead_Eval_Local --model gpt-35-turbo --tasks dp --log_path results/planning_gpt35_lookahead_local_20_5 --data_path data/dp/data_no_scratchpad_n_6_minval_-5_maxval_5_sampled_1000.jsonl --prompt_path agentboard/prompts/Planning/dp_lookahead_light.json --lookahead_length 5

# gsm8k
python agentboard/eval_planning.py --cfg-path eval_configs/dp/planning_results_all_tasks.yaml --algorithm Lookahead_Eval_Local --model gpt-35-turbo --tasks dp --log_path results/planning_gpt35_lookahead_local_20_6 --data_path data/dp/data_no_scratchpad_n_6_minval_-5_maxval_5_sampled_1000.jsonl --prompt_path agentboard/prompts/Planning/dp_lookahead_light.json --lookahead_length 6