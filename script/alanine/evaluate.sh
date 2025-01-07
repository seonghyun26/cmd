cd ../../

CUDA_VISIBLE_DEVICES=$1 python main.py \
    --config-name contrastive-triplet \
    ++training.train=False \
    ++training.ckpt_file=1230-024411 \
    ++job.generate=False \
    ++job.traj_dir=/home/shpark/prj-cmd/cmd/outputs/2025-01-07/07-30-00/tps/0