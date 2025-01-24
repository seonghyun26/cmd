cd ../../

k_list=(1000 2000 3000 4000)
for i in "${!k_list[@]}"; do
    CUDA_VISIBLE_DEVICES=$1 python main.py \
        --config-name steered-gnncvtica \
        ++job.simulation.k=${k_list[$i]} 
    sleep 2
done


