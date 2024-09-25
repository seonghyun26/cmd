cd ../../

CUDA_VISIBLE_DEVICES=$1 python main.py \
    --config-name double-well \
    ++training.loss.name=mae \
    ++training.repeat=True