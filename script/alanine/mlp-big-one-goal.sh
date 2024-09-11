cd ../../

CUDA_VISIBLE_DEVICES=$1 python main.py \
    --config-name one-goal \
    ++training.loss.name=mse+reg