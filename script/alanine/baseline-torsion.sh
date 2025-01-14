cd ../../

k_list=(100 200 500 1000 2000 5000 10000 20000 50000 100000)
for i in "${!k_list[@]}"; do
    CUDA_VISIBLE_DEVICES=$1 python main.py \
        --config-name steered-torsion \
        ++job.simulation.k=${k_list[$i]}
    sleep 2
done

k_list=(100 200 500 1000 2000 5000 10000 20000 50000 100000)
for i in "${!k_list[@]}"; do
    CUDA_VISIBLE_DEVICES=$1 python main.py \
        --config-name steered-torsion \
        ++job.simulation.k=${k_list[$i]} \
        ++job.simulation.time_horizon=500
    sleep 2
done

