cd ../../

CUDA_VISIBLE_DEVICES=$1 python main.py \
    --config-name mlp \
    ++training.loss.name=mse+reg3 \
    ++training.noise_scale=0.1 \
    ++training.scale=1000.0 \
    ++data.temperature=500.0 \
    ++data.index=random-v2 \