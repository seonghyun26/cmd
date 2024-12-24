cd ../../

# batch_list=(4096 8192 16384 4096 8192 16384)
# sim_length_list=(1000 1000 1000 2000 2000 2000)

# for i in "${!batch_list[@]}"; do
#     CUDA_VISIBLE_DEVICES=$((i + 1)) python main.py \
#         --config-name contrastive \
#         ++training.loader.batch_size=${batch_list[$i]} \
#         ++job.simulation.time_horizon=${sim_length_list[$i]} &
#     sleep 2
# done

CUDA_VISIBLE_DEVICES=$1 python main.py \
    --config-name contrastive