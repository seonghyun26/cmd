cd ../../

CUDA_VISIBLE_DEVICES=$1 python main.py \
    --config-name contrastive-triplet \
    ++job.simulation.k=800000 \
    ++job.simulation.time_horizon=1000 \
    ++data.version=v7

# data_version_list=(v5 v6 v7)
# for i in "${!data_version_list[@]}"; do
#     CUDA_VISIBLE_DEVICES=$(($i + 4)) python main.py \
#         --config-name contrastive-triplet \
#         ++job.simulation.k=800000 \
#         ++job.simulation.time_horizon=1000 \
#         ++data.version=${data_version_list[$i]} &
#     sleep 2
# done

# k_list=(400000 700000 800000 900000)

# for i in "${!k_list[@]}"; do
#     CUDA_VISIBLE_DEVICES=$i python main.py \
#         --config-name contrastive-triplet \
#         ++job.simulation.k=${k_list[$i]} \
#         ++job.simulation.time_horizon=1000 \
#         ++training.train=False \
#         ++training.ckpt_file=1230-024411 &
#     sleep 2
#     CUDA_VISIBLE_DEVICES=$(($i + 4)) python main.py \
#         --config-name contrastive-triplet \
#         ++job.simulation.k=${k_list[$i]} \
#         ++job.simulation.time_horizon=2000 \
#         ++training.train=False \
#         ++training.ckpt_file=1230-024411 &
#     sleep 2
# done