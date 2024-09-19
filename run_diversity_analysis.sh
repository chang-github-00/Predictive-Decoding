#! /bin/bash
#
# run_diversity_analysis.sh
# Copyright (C) 2024-09-15 Junxian <He>
#
# Distributed under terms of the MIT license.
#

# python agentboard/calculate_generation_diversity.py --file_path results/ablation_study_humaneval_temp_0.6_0.01


# python agentboard/calculate_generation_diversity.py --file_path results/ablation_study_humaneval_temp_0.6_0.05

# python agentboard/calculate_generation_diversity.py --file_path results/run_parallel_passk_mpc_sample_humaneval_9_14_N_10

# python agentboard/calculate_generation_diversity.py --file_path results/ablation_study_humaneval_temp_0.6_0.5

# python agentboard/calculate_generation_diversity.py --file_path results/ablation_study_humaneval_temp_0.6_1.0

# python agentboard/calculate_generation_diversity.py --file_path results/ablation_study_humaneval_temp_0.6_5.0

# python agentboard/calculate_generation_diversity.py --file_path results/run_parallel_humaneval_llama3_passk_9_14_beamsearch

python agentboard/calculate_generation_diversity.py --file_path results/ablation_study_humaneval_temp_0.1_0.05
python agentboard/calculate_generation_diversity.py --file_path results/ablation_study_humaneval_temp_0.3_0.05
python agentboard/calculate_generation_diversity.py --file_path results/ablation_study_humaneval_temp_0.8_0.05
python agentboard/calculate_generation_diversity.py --file_path results/ablation_study_humaneval_temp_1.0_0.05


