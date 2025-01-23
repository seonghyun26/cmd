cd ../../

# CUDA_VISIBLE_DEVICES=$1 python main.py \
#     --config-name steered-deeptica

k_list=(400 500 600 700 800 900)

for i in "${!k_list[@]}"; do
    CUDA_VISIBLE_DEVICES=$1 python main.py \
        --config-name steered-deeptica \
        ++training.ckpt_file=timelag-1n-v1 \
        ++job.simulation.k=${k_list[$i]} 
    sleep 2
done

for i in "${!k_list[@]}"; do
    CUDA_VISIBLE_DEVICES=$1 python main.py \
        --config-name steered-deeptica \
        ++training.ckpt_file=timelag-10n-v1 \
        ++job.simulation.k=${k_list[$i]} 
    sleep 2
done

for i in "${!k_list[@]}"; do
    CUDA_VISIBLE_DEVICES=$1 python main.py \
        --config-name steered-deeptica \
        ++training.ckpt_file=timelag-250n-v1 \
        ++job.simulation.k=${k_list[$i]} 
    sleep 2
done