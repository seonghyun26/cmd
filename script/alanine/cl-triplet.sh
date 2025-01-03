cd ../../

CUDA_VISIBLE_DEVICES=$1 python main.py \
    --config-name contrastive-triplet \
    ++data.version=v4 

# batch_list=(2048 4096 8192 16384)
# for i in "${!batch_list[@]}"; do
#     CUDA_VISIBLE_DEVICES=$((i + 3)) python main.py \
#         --config-name contrastive-triplet \
#         ++training.loader.batch_size=${batch_list[$i]} &
#     sleep 2
# done