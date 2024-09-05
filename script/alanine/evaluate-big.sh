cd ../../

CUDA_VISIBLE_DEVICES=$1 python main.py \
    --config-name mlp-big \
    model=mlp-big \
    ++training.train=False \
    ++training.ckpt_name=16-layer3 \
    job.metrics="['epd','thp','ram']"
