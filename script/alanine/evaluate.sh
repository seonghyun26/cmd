cd ../../

CUDA_VISIBLE_DEVICES=$1 python main.py \
    --config-name mlp-big \
    model=mlp-big \
    ++training.train=False \
    ++training.ckpt_name=mlp-big \
    job.metrics="['epd','thp','ram']"
