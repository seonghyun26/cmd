cd ../../

CUDA_VISIBLE_DEVICES=$1 python main.py \
    --config-name double-well \
    ++training.loss.name=mse+reg3 \
    ++training.repeat=True \
    ++data.temperature=mix-both \
    ++data.state=both \
    ++data.index=multi-temp-v3