cd ../../

CUDA_VISIBLE_DEVICES=$1 python main.py \
    --config-name mlp \
    ++model.transform=ic4 \
    ++training.loss.name=mse+reg5 \
    ++training.noise_scale=0.4 \
    ++data.temperature=500.0 \
    ++data.version=random-v2 \