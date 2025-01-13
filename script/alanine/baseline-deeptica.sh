cd ../../

# CUDA_VISIBLE_DEVICES=$1 python main.py \
#     --config-name steered-deeptica



k_list=(1000 1500 2000 2500 3000)
for i in "${!k_list[@]}"; do
    CUDA_VISIBLE_DEVICES=$1 python main.py \
        --config-name steered-deeptica \
        ++training.ckpt_file=deeptica-v2 \
        ++job.simulation.k=${k_list[$i]}
    sleep 2
done

k_list=(2500 3000 3500 4000 4500)
for i in "${!k_list[@]}"; do
    CUDA_VISIBLE_DEVICES=$1 python main.py \
        --config-name steered-deeptica \
        ++training.ckpt_file=deeptica-v2 \
        ++job.simulation.time_horizon=500 \
        ++job.simulation.k=${k_list[$i]}
    sleep 2
done

k_list=(1000 2000 3000 4000)
for i in "${!k_list[@]}"; do
    CUDA_VISIBLE_DEVICES=$1 python main.py \
        --config-name steered-deeptica \
        ++training.ckpt_file=deeptica-v4 \
        ++job.simulation.k=${k_list[$i]}
    sleep 2
done

k_list=(1000 2000 3000 4000)
for i in "${!k_list[@]}"; do
    CUDA_VISIBLE_DEVICES=$1 python main.py \
        --config-name steered-deeptica \
        ++training.ckpt_file=deeptica-v4 \
        ++job.simulation.time_horizon=500 \
        ++job.simulation.k=${k_list[$i]}
    sleep 2
done