cd ../../

# CUDA_VISIBLE_DEVICES=$1 python main.py \
#     --config-name contrastive-triplet \
#     ++training.train=False \
#     ++training.ckpt_file=1230-024411 \

k_list=(1000000 11000000 12000000 13000000) 
for i in "${!k_list[@]}"; do
    CUDA_VISIBLE_DEVICES=$1 python main.py \
        --config-name contrastive-triplet \
        ++training.train=False \
        ++training.ckpt_file=1230-024411 \
        ++job.simulation.k=${k_list[$i]} \
        ++job.simulation.time_horizon=500
        # ++training.ckpt_file=250113-024542-300 \
    sleep 2
done
