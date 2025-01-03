cd ../../

CUDA_VISIBLE_DEVICES=$1 python main.py \
    --config-name steered-aecv2

# for k in 400 300 200 100; do
#     for sim_length in 500 1000; do
#         CUDA_VISIBLE_DEVICES=$1 python main.py \
#             --config-name steered-aecv2 \
#             ++job.simulation.k=$k \
#             ++job.simulation.time_horizon=$sim_length
#     done
# done