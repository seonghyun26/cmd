cd ../../

for batch_size in 2048 4096 8192; do
    for k in 1000 10000 100000; do
        CUDA_VISIBLE_DEVICES=$1 python main.py \
            --config-name contrastive-triplet \
            ++training.loader.batch_size=$batch_size \
            ++job.simulation.k=$k
    done
done