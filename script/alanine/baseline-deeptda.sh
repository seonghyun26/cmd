cd ../../

# CUDA_VISIBLE_DEVICES=$1 python main.py \
#     --config-name steered-deeptda


k_list=(400 500 600 700 800 900)

for i in "${!k_list[@]}"; do
    CUDA_VISIBLE_DEVICES=$1 python main.py \
        --config-name steered-deeptda \
        ++training.ckpt_file=da-1n-v1 \
        ++job.simulation.k=${k_list[$i]} 
    sleep 2
done

for i in "${!k_list[@]}"; do
    CUDA_VISIBLE_DEVICES=$1 python main.py \
        --config-name steered-deeptda \
        ++training.ckpt_file=da-10n-v1 \
        ++job.simulation.k=${k_list[$i]} 
    sleep 2
done

for i in "${!k_list[@]}"; do
    CUDA_VISIBLE_DEVICES=$1 python main.py \
        --config-name steered-deeptda \
        ++training.ckpt_file=da-250n-v1 \
        ++job.simulation.k=${k_list[$i]} 
    sleep 2
done