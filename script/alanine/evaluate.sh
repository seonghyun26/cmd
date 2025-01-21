cd ../../

CUDA_VISIBLE_DEVICES=$1 python main.py \
    --config-name test-cl \
    ++training.train=False \
    ++training.ckpt_file=250121-0704 \
    ++job.simulation.k=1000

# k_list=(800000) 
# for i in "${!k_list[@]}"; do
#     CUDA_VISIBLE_DEVICES=$1 python main.py \
#         --config-name contrastive-triplet \
#         ++training.train=False \
#         ++training.ckpt_file=1230-024411 \
#         ++job.simulation.k=${k_list[$i]} 
#         # ++job.simulation.time_horizon=500
#         # ++training.ckpt_file=250113-024542-300 \
#     sleep 2
# done
