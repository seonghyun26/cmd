cd ../../

k_list=(600000 800000 1000000)
for i in "${!k_list[@]}"; do
    CUDA_VISIBLE_DEVICES=2 python main.py \
        --config-name contrastive-triplet \
        ++job.simulation.k=${k_list[$i]} \
        ++data.version=v5
        # ++training.train=False \
        # ++training.ckpt_file=ablation-cl-v5
    sleep 2
done
