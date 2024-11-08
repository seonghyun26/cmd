cd ../../

for batch_size in 2048 4096 8192; do
    for k in 1000 10000 100000; do
        for dim in 2 4 8; do
            CUDA_VISIBLE_DEVICES=$1 python main.py \
                --config-name contrastive-nce \
                ++training.loader.batch_size=$batch_size \
                ++job.simulation.k=$k \
                ++model.params.output_dim=$dim
        done
    done
done