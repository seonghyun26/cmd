cd ../../

CUDA_VISIBLE_DEVICES=$1 python main.py \
    --config-name mlp-big \
    ++training.loss.name=mse+reg