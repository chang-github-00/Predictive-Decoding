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


python agentboard/eval_reasoning.py --cfg-path eval_configs/gsm8k/mpc_sample_gsm8k_llama3.yaml --tasks gsm8k --algorithm MPC_Sample --model llama3 --data_path /root/huggingface/gsm8k --log_path results/run_mpc_gsm8k_llama3_7_13
python agentboard/eval_reasoning.py --cfg-path eval_configs/gsm8k/self_consistency_gsm8k_llama3.yaml --tasks gsm8k --algorithm Self_Consistency --model llama-3 --data_path /root/huggingface/gsm8k --log_path results/run_self_consistency_gsm8k_llama3_8_1_n_1_pal_prompt
python agentboard/eval_reasoning.py --cfg-path eval_configs/gsm8k/mpc_sample_gsm8k_llama3.yaml --tasks gsm8k --algorithm MPC_Sample --model llama-3 --data_path /root/huggingface/gsm8k --log_path results/run_mpc_gsm8k_llama3_8_2_pal_prompt_memory
python agentboard/eval_reasoning.py --cfg-path eval_configs/math/self_consistency_math_llama3.yaml --tasks math --algorithm Self_Consistency --model llama-3 --data_path /root/Agent-Decoding/data/math/test.json --log_path results/run_self_consistency_math_llama3_8_7_pal_prompt
python agentboard/eval_reasoning.py --cfg-path eval_configs/math/mpc_sample_math_llama3.yaml --tasks math --algorithm MPC_Sample --model llama-3 --data_path /root/Agent-Decoding/data/math/test.json --log_path results/run_mpc_sample_math_llama3_8_8_pal_prompt


# math parallel run pal, set n generate samples to 1
python agentboard/eval_reasoning_parallel.py --cfg-path eval_configs_hkust_server/math/self_consistency_math_llama3.yaml --tasks math --algorithm Self_Consistency --model llama-3 --data_path /ssddata/junxianh/Agent-Decoding/data/math/test.json --log_path results/run_parallel_self_consistency_math_8_21_n_8 --batch_size 5000
python agentboard/eval_reasoning_parallel.py --cfg-path eval_configs_hkust_server/gsm8k/self_consistency_gsm8k_llama3.yaml --tasks gsm8k --algorithm Self_Consistency --model llama-3 --data_path /ssddata/junxianh/Agent-Decoding/data/gsm8k --log_path results/run_parallel_self_consistency_gsm8k_8_21_n_8 --batch_size 1319
python agentboard/eval_reasoning_parallel.py --cfg-path eval_configs_hkust_server/gsm8k/self_consistency_gsm8k_llama3.yaml --tasks gsm8k --algorithm Self_Consistency --model llama-3 --data_path /ssddata/junxianh/Agent-Decoding/data/gsm8k --log_path results/run_parallel_self_consistency_gsm8k_8_25_n_8_beam_search --batch_size 1319
python agentboard/eval_reasoning_parallel.py --cfg-path eval_configs_hkust_server/math/self_consistency_math_llama3.yaml --tasks math --algorithm Self_Consistency --model llama-3 --data_path /ssddata/junxianh/Agent-Decoding/data/math/test.json --log_path results/run_parallel_self_consistency_math_8_25_n_8_beam_search --batch_size 5000


# math parallel run predictive decoding, set n generate samples to 1
python agentboard/eval_reasoning_parallel.py --cfg-path eval_configs/math/mpc_sample_math_llama3.yaml --tasks math --algorithm MPC_Sample --model llama-3 --data_path /ssddata/junxianh/Agent-Decoding/data/math/test.json --log_path results/run_parallel_mpc_sample_math_8_21 --batch_size 500
python agentboard/eval_reasoning_parallel.py --cfg-path eval_configs_hkust_server/math/mpc_sample_math_llama3.yaml --tasks math --algorithm MPC_Sample --model llama-3 --data_path /ssddata/junxianh/Agent-Decoding/data/math/test.json --log_path results/run_parallel_mpc_sample_math_8_30 --batch_size 500

python agentboard/eval_reasoning_parallel.py --cfg-path eval_configs_hkust_server/gsm8k/mpc_sample_gsm8k_llama3.yaml --tasks gsm8k --algorithm MPC_Sample --model llama-3 --data_path /ssddata/junxianh/Agent-Decoding/data/gsm8k --log_path results/run_parallel_mpc_sample_gsm8k_8_21 --batch_size 1319


# humaneval parallel run, set n generate samples to 10
python agentboard/eval_humaneval_parallel.py --cfg-path eval_configs_hkust_server/humaneval/self_consistency_humaneval_llama3.yaml --tasks humaneval --algorithm Self_Consistency --model llama-3 --data_path /ssddata/junxianh/Agent-Decoding/data/humaneval/humaneval-python.jsonl --log_path results/run_parallel_passk_humaneval_8_23 --batch_size 200

python agentboard/eval_reasoning_parallel.py --cfg-path eval_configs_hkust_server/math/self_consistency_math_llama3.yaml --tasks math --algorithm Self_Consistency --model llama-3 --data_path /ssddata/junxianh/Agent-Decoding/data/math/test.json --log_path results/run_parallel_self_consistency_math_8_28_n_8_beam_search --batch_size 5000


python agentboard/eval_reasoning_parallel.py --cfg-path eval_configs_hkust_server/gsm8k/mpc_multiple_gsm8k_llama3.yaml --tasks gsm8k --algorithm MPC_Sample --model llama-3 --data_path /ssddata/junxianh/Agent-Decoding/data/gsm8k --log_path results/run_parallel_mpc_sample_gsm8k_8_28_n_8 --batch_size 50
python agentboard/eval_reasoning_parallel.py --cfg-path eval_configs_hkust_server/math/mpc_multiple_math_llama3.yaml --tasks math --algorithm MPC_Sample --model llama-3 --data_path /ssddata/junxianh/Agent-Decoding/data/math/test.json --log_path results/run_parallel_mpc_sample_math_8_28_n_8 --batch_size 10


python agentboard/eval_reasoning_parallel.py --cfg-path eval_configs_hkust_server/gsm8k/self_consistency_gsm8k_llama3.yaml --tasks gsm8k --algorithm Self_Consistency --model llama-3 --data_path /ssddata/junxianh/Agent-Decoding/data/gsm8k --log_path results/run_parallel_self_consistency_gsm8k_8_31_n_8_sample --batch_size 1319

# mbpp parallel run, set n generate samples to 10
python agentboard/eval_humaneval_parallel.py --cfg-path eval_configs_hkust_server/mbpp/self_consistency_mbpp_llama3.yaml --tasks mbpp --model llama-3 --log_path results/run_self_consistency_mbpp_9_6 --algorithm Self_Consistency --batch_size 500 --data_path /ssddata/junxianh/Agent-Decoding/data/mbpp/mbpp_test.jsonl

# mbpp mpc sample parallel run, set n generate samples to 1
python agentboard/eval_humaneval_parallel.py --cfg-path eval_configs_hkust_server/mbpp/mpc_sample_mbpp_llama3.yaml --tasks mbpp --model llama-3 --log_path results/run_mpc_sample_n_1_mbpp_9_6 --algorithm MPC_Sample --batch_size 500 --data_path /ssddata/junxianh/Agent-Decoding/data/mbpp/mbpp_test.jsonl
# mbpp mpc sample parallel run, set n generate samples to 10
python agentboard/eval_humaneval_parallel.py --cfg-path eval_configs_hkust_server/mbpp/mpc_multiple_mbpp_llama3.yaml --tasks mbpp --model llama-3 --log_path results/run_mpc_sample_n_10_mbpp_9_6 --algorithm MPC_Sample --batch_size 500 --data_path /ssddata/junxianh/Agent-Decoding/data/mbpp/mbpp_test.jsonl
