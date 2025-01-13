cd ../../

# CUDA_VISIBLE_DEVICES=$1 python main.py \
#     --config-name contrastive-triplet \
#     ++training.train=False \
#     ++training.ckpt_file=1230-024411 \

k_list=(900000 1000000)
for i in "${!k_list[@]}"; do
    CUDA_VISIBLE_DEVICES=$((i + 6)) python main.py \
        --config-name contrastive-triplet \
        ++training.train=False \
        ++training.ckpt_file=250111-121037 \
        ++job.simulation.k=${k_list[$i]} &
    sleep 2
done
