cd ../../

batch_list=(256 512 1024 2048 4096 8192)

for i in "${!batch_list[@]}"; do
    CUDA_VISIBLE_DEVICES=$((i + 1)) python main.py \
        --config-name contrastive-triplet \
        ++training.loader.batch_size=${batch_list[$i]} &
        sleep 2
done