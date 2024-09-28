export CUDA_VISIBLE_DEVICES=2
export OPENAI_API_BASE="http://localhost:8001/v1"


# python agentboard/eval_reasoning_reward_parallel.py --cfg-path eval_configs/scaling_law/mpc/mpc_k_2.yaml --tasks gsm8k --algorithm MPC_Sample_Reward --model llama-3 --data_path /root/huggingface/gsm8k --log_path results/scaling_law_experiment_mpc_k_2_error_bar --batch_size 200 --reward_model math-shepherd
# python agentboard/eval_reasoning_reward_parallel.py --cfg-path eval_configs/scaling_law/mpc/mpc_k_4.yaml --tasks gsm8k --algorithm MPC_Sample_Reward --model llama-3 --data_path /root/huggingface/gsm8k --log_path results/scaling_law_experiment_mpc_k_4_error_bar --batch_size 200 --reward_model math-shepherd


# python agentboard/eval_reasoning_reward_parallel.py --cfg-path eval_configs/scaling_law/mpc/mpc_t_2.yaml --tasks gsm8k --algorithm MPC_Sample_Reward --model llama-3 --data_path /root/huggingface/gsm8k --log_path results/scaling_law_experiment_mpc_k_8_t_2_error_bar --batch_size 200 --reward_model math-shepherd
# python agentboard/eval_reasoning_reward_parallel.py --cfg-path eval_configs/scaling_law/mpc/mpc_t_4.yaml --tasks gsm8k --algorithm MPC_Sample_Reward --model llama-3 --data_path /root/huggingface/gsm8k --log_path results/scaling_law_experiment_mpc_k_8_t_4_error_bar --batch_size 200 --reward_model math-shepherd

# python agentboard/eval_reasoning_reward_parallel.py --cfg-path eval_configs/scaling_law/mpc/mpc_k_1.yaml --tasks gsm8k --algorithm MPC_Sample_Reward --model llama-3 --data_path /root/huggingface/gsm8k --log_path results/scaling_law_experiment_mpc_k_2_error_bar --batch_size 20 --reward_model math-shepherd

python agentboard/eval_reasoning_reward_parallel.py --cfg-path eval_configs/scaling_law/mpc/mpc_t_2.yaml --tasks gsm8k --algorithm MPC_Sample_Reward --model llama-3 --data_path /root/huggingface/gsm8k --log_path results/scaling_law_experiment_mpc_k_2_t_2_error_bar --batch_size 200 --reward_model math-shepherd
