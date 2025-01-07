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

k_list=(400 600 800)
for i in "${!k_list[@]}"; do
    CUDA_VISIBLE_DEVICES=$1 python main.py \
        --config-name steered-deeplda \
        ++job.simulation.k=${k_list[$i]}
    sleep 2
done