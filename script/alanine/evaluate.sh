cd ../../

CUDA_VISIBLE_DEVICES=$1 python main.py \
    --config-name steered-clcv

# k_list=(3000 4000 5000 6000 7000 8000) 
# for i in "${!k_list[@]}"; do
#     CUDA_VISIBLE_DEVICES=$1 python main.py \
#         --config-name steered-clcv \
#         ++training.train=False \
#         ++training.ckpt_file=250121-0704 \
#         ++job.simulation.k=${k_list[$i]} 
#     sleep 2
# done
