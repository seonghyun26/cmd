cd ../../

# CUDA_VISIBLE_DEVICES=$1 python main.py \
#     --config-name steered-deeptda

# for k in 100 80 60 40 20; do
#     for sim_length in 500 1000 2000 4000; do
#         CUDA_VISIBLE_DEVICES=$1 python main.py \
#             --config-name steered-deeptda \
#             ++job.simulation.k=$k \
#             ++job.simulation.time_horizon=$sim_length
#     done
# done



# k_list=(110 120 130 140 150 160)
# for i in "${!k_list[@]}"; do
#     CUDA_VISIBLE_DEVICES=$1 python main.py \
#         --config-name steered-deeptda \
#         ++job.simulation.k=${k_list[$i]} \
#         ++training.ckpt_file=deeptda-v1
#     sleep 2
# done

k_list=(180 160 140 120 100 300 320 340 360)
for i in "${!k_list[@]}"; do
    CUDA_VISIBLE_DEVICES=$1 python main.py \
        --config-name steered-deeptda \
        ++job.simulation.k=${k_list[$i]}
    sleep 2
done


# k_list=(100 120 140 160 180 200 300 400)
# for i in "${!k_list[@]}"; do
#     CUDA_VISIBLE_DEVICES=$1 python main.py \
#         --config-name steered-deeptda \
#         ++job.simulation.k=${k_list[$i]} \
#         ++job.simulation.time_horizon=500 \
#         ++training.ckpt_file=deeptda-v1
#     sleep 2
# done

k_list=(100 120 140 160 180 200)
for i in "${!k_list[@]}"; do
    CUDA_VISIBLE_DEVICES=$1 python main.py \
        --config-name steered-deeptda \
        ++job.simulation.k=${k_list[$i]} \
        ++job.simulation.time_horizon=500 \
        ++training.ckpt_file=deeptda-v2
    sleep 2
done