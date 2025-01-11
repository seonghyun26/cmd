cd ../../

# CUDA_VISIBLE_DEVICES=$1 python main.py \
#     --config-name steered-deeplda \
#     ++job.simulation.k=1000 \
#     ++job.simulation.time_horizon=1000

# for k in 4000 3000 2000 1000 500; do
#     for sim_length in 500 1000 2000 4000; do
#         CUDA_VISIBLE_DEVICES=$1 python main.py \
#             --config-name steered-deeplda \
#             ++job.simulation.k=$k \
#             ++job.simulation.time_horizon=$sim_length
#     done
# done

k_list=(10 50 100 200 400 600)
for i in "${!k_list[@]}"; do
    CUDA_VISIBLE_DEVICES=$1 python main.py \
        --config-name steered-deeplda \
        ++job.simulation.k=${k_list[$i]} \
        ++job.simulation.time_horizon=500 \
        ++training.ckpt_file=deeplda-v1 
    sleep 2
done

k_list=(10 50 100 200 400 600)
for i in "${!k_list[@]}"; do
    CUDA_VISIBLE_DEVICES=$1 python main.py \
        --config-name steered-deeplda \
        ++job.simulation.k=${k_list[$i]} \
        ++job.simulation.time_horizon=500 \
        ++training.ckpt_file=deeplda-v2 
    sleep 2
done