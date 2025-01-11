cd ../../

# CUDA_VISIBLE_DEVICES=$1 python main.py \
#     --config-name steered-ae

# for k in 400 300 200 100; do
#     for sim_length in 500 1000; do
#         CUDA_VISIBLE_DEVICES=$1 python main.py \
#             --config-name steered-aecv \
#             ++job.simulation.k=$k \
#             ++job.simulation.time_horizon=$sim_length
#     done
# done

# k_list=(100 200 300)
# for i in "${!k_list[@]}"; do
#     CUDA_VISIBLE_DEVICES=$1 python main.py \
#         --config-name steered-ae \
#         ++job.simulation.k=${k_list[$i]} \
#         ++training.ckpt_file=aecv-v2
#     sleep 2
# done

k_list=(10 100 200 400 600 800 1000)
for i in "${!k_list[@]}"; do
    CUDA_VISIBLE_DEVICES=$1 python main.py \
        --config-name steered-ae \
        ++job.simulation.k=${k_list[$i]} \
        ++job.simulation.time_horizon=500 \
        ++training.ckpt_file=aecv-v3 
    sleep 2
done

k_list=(10 100 200 400 600 800 1000)
for i in "${!k_list[@]}"; do
    CUDA_VISIBLE_DEVICES=$1 python main.py \
        --config-name steered-ae \
        ++job.simulation.k=${k_list[$i]} \
        ++job.simulation.time_horizon=500 \
        ++training.ckpt_file=aecv-v3 
    sleep 2
done