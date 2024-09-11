cd ../../

CUDA_VISIBLE_DEVICES=$1 python main.py \
    --config-name mlp-big \
    ++training.loss.name=mse+reg3 \
    ++training.noise_scale=1 \
    ++training.scale=1000.0 \
    ++data.temperature=500.0 \
    ++data.index=random-v1 \