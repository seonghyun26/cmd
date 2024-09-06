cd ../../

CUDA_VISIBLE_DEVICES=$1 python main.py \
    --config-name mlp-big \
    ++data.index=random-v2   