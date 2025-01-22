cd ../../

# CUDA_VISIBLE_DEVICES=$1 python main.py \
#     --config-name test-cl \
#     ++job.sample_num=10 \
#     ++job.simulation.k=1000

k_list=(1200 1400 1600 1800 2000)
for i in "${!k_list[@]}"; do
    CUDA_VISIBLE_DEVICES=$1 python main.py \
        --config-name test-cl \
        ++job.sample_num=10 \
        ++job.simulation.k=${k_list[$i]} \
        ++training.train=False \
        ++training.ckpt_file=250121-091350
    sleep 2
done

# batch_list=(4096 8192 16384)
# for i in "${!batch_list[@]}"; do
#     CUDA_VISIBLE_DEVICES=$((i + 2)) python main.py \
#         --config-name debug-cl \
#         ++training.loader.batch_size=${batch_list[$i]} &
#     sleep 2
# done