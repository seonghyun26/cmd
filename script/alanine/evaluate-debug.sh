cd ../../

CUDA_VISIBLE_DEVICES=$1 python main.py \
    --config-name debug \
    ++training.train=False \
    ++training.ckpt_name=3-layer
