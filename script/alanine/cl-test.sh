cd ../../

CUDA_VISIBLE_DEVICES=$1 python main.py \
    --config-name test-cl \
    ++job.sample_num=10

# k_list=(4000 5000 6000 7000 8000)
# for i in "${!k_list[@]}"; do
#     CUDA_VISIBLE_DEVICES=$1 python main.py \
#         --config-name test-cl \
#         ++job.sample_num=10 \
#         ++job.simulation.k=${k_list[$i]}
#     sleep 2
# done

# batch_list=(4096 8192 16384)
# for i in "${!batch_list[@]}"; do
#     CUDA_VISIBLE_DEVICES=$((i + 2)) python main.py \
#         --config-name debug-cl \
#         ++training.loader.batch_size=${batch_list[$i]} &
#     sleep 2
# done