cd ../../

k_list=(3000 4000 5000 6000 7000 8000)
for i in "${!k_list[@]}"; do
    CUDA_VISIBLE_DEVICES=$1 python main.py \
        --config-name steered-rmsd \
        ++job.simulation.k=${k_list[$i]}
    sleep 2
done

k_list=(3000 4000 5000 6000 7000 8000)
for i in "${!k_list[@]}"; do
    CUDA_VISIBLE_DEVICES=$1 python main.py \
        --config-name steered-rmsd \
        ++job.simulation.k=${k_list[$i]} \
        ++job.simulation.time_horizon=500
    sleep 2
done

