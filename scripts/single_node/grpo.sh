# 4 GPU, train_sd3, use gradient, choose whether to eval ode_log_prob in config
accelerate launch --config_file scripts/accelerate_configs/multi_gpu.yaml --num_processes=4 --main_process_port 29502 scripts/train_sd3.py --config config/grpo.py:pickscore_sd3_1batch_size_use_gradient
# 4 GPU, train_sd3, without using gradient
# accelerate launch --config_file scripts/accelerate_configs/multi_gpu.yaml --num_processes=4 --main_process_port 29501 scripts/train_sd3.py --config config/grpo.py:pickscore_sd3_1batch_size
# 4 GPU, train_sd3_fast
# accelerate launch --config_file scripts/accelerate_configs/multi_gpu.yaml --num_processes=4 --main_process_port 29501 scripts/train_sd3_fast.py --config config/grpo.py:pickscore_sd3_fast_4gpu
# 4 GPU, train_sd3_ode_logprob
# accelerate launch --config_file scripts/accelerate_configs/multi_gpu.yaml --num_processes=4 --main_process_port 29501 scripts/train_sd3_ode_logprob.py --config config/grpo.py:pickscore_sd3_4gpu_ode_logprob