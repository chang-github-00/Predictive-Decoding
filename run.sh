CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 python agentboard/eval_main.py --cfg-path eval_configs/react_results_all_tasks.yaml --model deepseek-67b --tasks alfworld pddl --log_path results/deepseek_react