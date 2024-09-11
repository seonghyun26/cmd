cd ../../

CUDA_VISIBLE_DEVICES=$1 python main.py \
    --config-name multi-next \
    ++job.temperature=500 \
    ++training.noise_scale=0.01