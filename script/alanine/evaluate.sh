cd ../../

# CUDA_VISIBLE_DEVICES=$1 python main.py \
#     --config-name contrastive-triplet \
#     ++training.train=False \
#     ++training.ckpt_file=1230-024411 \

k_list=(800000 900000 1000000)
for i in "${!k_list[@]}"; do
    CUDA_VISIBLE_DEVICES=$((i + 5)) python main.py \
        --config-name contrastive-triplet \
        ++training.train=False \
        ++training.ckpt_file=1230-024411 \
        ++job.simulation.k=${k_list[$i]} &
        # ++training.ckpt_file=250113-024542-300 \
    sleep 2
done
