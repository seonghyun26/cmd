cd ../../

CUDA_VISIBLE_DEVICES=$1 python main.py \
    --config-name mlp-huge \
    ++training.loss.name=mse+reg3 \
    ++training.noise_scale=0.1
