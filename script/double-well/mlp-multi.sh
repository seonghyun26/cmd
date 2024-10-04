cd ../../
for i in {1..3}
do
    noise_values=(0.02 0.04 0.08)
    noise=${noise_values[$((i-1))]}
    for j in {1..2}
    do
        horizon=$((300 * j))
        CUDA_VISIBLE_DEVICES=$1 python main.py \
            --config-name double-well \
            ++training.loss.name=mae+reg4 \
            ++training.repeat=True \
            ++training.noise_scale=$noise \
            ++job.time_horizon=$horizon \
            ++data.temperature=mix-both \
            ++data.state=both \
            ++data.index=multi-temp-v3
        sleep 1
    done
done