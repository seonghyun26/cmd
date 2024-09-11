cd ../../

CUDA_VISIBLE_DEVICES=$1 python main.py \
    --config-name mlp \
    ++training.loss.name=mse+reg4 \
    ++training.noise_scale=1 \
    ++training.scale=1000.0 \
    ++data.temperature=500.0 \
    ++data.index=random-v2 \