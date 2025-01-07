cd ../../

data_version_list=(v5 v6 v7 v8)
for i in "${!data_version_list[@]}"; do
    CUDA_VISIBLE_DEVICES=$(($i + 4)) python main.py \
        --config-name contrastive-triplet \
        ++job.simulation.k=800000 \
        ++job.simulation.time_horizon=1000 \
        ++data.version=${data_version_list[$i]} &
    sleep 2
done
