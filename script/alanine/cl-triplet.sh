cd ../../

batch_list=(256 512 1024 2048 4096 8192)
# sim_length_list=(1000 1000 1000)

for i in "${!batch_list[@]}"; do
    CUDA_VISIBLE_DEVICES=$((i + 1)) python main.py \
        --config-name contrastive-triplet \
        ++data.augmentation=hard \
        ++training.loader.batch_size=${batch_list[$i]} \
        ++job.simulation.time_horizon=1000 \
        ++ &
        # ++job.simulation.time_horizon=${sim_length_list[$i]} &
    sleep 2
done