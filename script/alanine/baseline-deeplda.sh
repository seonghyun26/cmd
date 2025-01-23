cd ../../

# CUDA_VISIBLE_DEVICES=$1 python main.py \
#     --config-name steered-deeplda \
#     ++job.simulation.k=1000 \
#     ++job.simulation.time_horizon=1000


k_list=(450 550)

for i in "${!k_list[@]}"; do
    CUDA_VISIBLE_DEVICES=$1 python main.py \
        --config-name steered-deeplda \
        ++training.ckpt_file=da-1n-v1 \
        ++job.simulation.k=${k_list[$i]} 
    sleep 2
done

for i in "${!k_list[@]}"; do
    CUDA_VISIBLE_DEVICES=$1 python main.py \
        --config-name steered-deeplda \
        ++training.ckpt_file=da-10n-v1 \
        ++job.simulation.k=${k_list[$i]} 
    sleep 2
done


k_list=(150 250 300 350 400)

for i in "${!k_list[@]}"; do
    CUDA_VISIBLE_DEVICES=$1 python main.py \
        --config-name steered-deeplda \
        ++training.ckpt_file=da-250n-v1 \
        ++job.simulation.k=${k_list[$i]} 
    sleep 2
done