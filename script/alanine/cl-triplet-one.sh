cd ../../

# CUDA_VISIBLE_DEVICES=$1 python main.py \
#     --config-name contrastive-triplet

k_list=(100 1000 10000 100000)
for i in "${!k_list[@]}"; do
    CUDA_VISIBLE_DEVICES=2 python main.py \
        --config-name contrastive-triplet-one \
        ++job.simulation.k=${k_list[$i]}
    sleep 2
done

# batch_list=(2048 4096 8192 16384)
# for i in "${!batch_list[@]}"; do
#     CUDA_VISIBLE_DEVICES=$((i + 3)) python main.py \
#         --config-name contrastive-triplet \
#         ++training.loader.batch_size=${batch_list[$i]} &
#     sleep 2
# done