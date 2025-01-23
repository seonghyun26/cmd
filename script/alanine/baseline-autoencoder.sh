cd ../../

# CUDA_VISIBLE_DEVICES=$1 python main.py \
#     --config-name steered-autoencoder

k_list=(300 400 500 600 700 800)

for i in "${!k_list[@]}"; do
    CUDA_VISIBLE_DEVICES=$1 python main.py \
        --config-name steered-autoencoder \
        ++training.ckpt_file=da-1n-v1 \
        ++job.simulation.k=${k_list[$i]} \
        ++job.cv_min=-3.371861457824707 \
        ++job.cv_max=2.55740618705749
    sleep 2
done

for i in "${!k_list[@]}"; do
    CUDA_VISIBLE_DEVICES=$1 python main.py \
        --config-name steered-autoencoder \
        ++training.ckpt_file=da-10n-v1 \
        ++job.simulation.k=${k_list[$i]} \
        ++job.cv_min=3.0287108421325684 \
        ++job.cv_max=2.723254919052124
    sleep 2
done

for i in "${!k_list[@]}"; do
    CUDA_VISIBLE_DEVICES=$1 python main.py \
        --config-name steered-autoencoder \
        ++training.ckpt_file=da-250n-v1 \
        ++job.simulation.k=${k_list[$i]}  \
        ++job.cv_min=-2.725149631500244 \
        ++job.cv_max=2.7825794219970703
    sleep 2
done