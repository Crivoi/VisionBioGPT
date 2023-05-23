#!/bin/bash

module load intel-mkl/2020.3.304
module load python3/3.9.2
module load cuda/11.2.2
module load cudnn/8.1.1-cuda11
module load openmpi/4.1.0
module load magma/2.6.0
module load fftw3/3.3.8
module load pytorch/1.9.0

data_dir=/checkpoints
dataset=mimic50

seed=52
length=1024
output=${dataset}_length_${length}_seed_${seed}
output_dir=${data_dir}/output/${output}_$(date +%F-%H-%M-%S-%N)

if ! test -f "../results/1/test/${output}.json"; then
  python3 train.py \
    --task_name multilabel \
    --output_metrics_filepath /checkpoints/metrics/metrics_train_${output}.json \
    --model_dir $data_dir/BioGpt/mimic_biogpt_base \
    --seed $seed \
    --train_filepath /data/50/train.json \
    --dev_filepath /data/50/dev.json \
    --output_dir $output_dir \
    --per_device_train_batch_size 2 \
    --gradient_accumulation_steps 8 \
    --learning_rate 2e-5 \
    --num_train_epochs 1 \
    --save_strategy epoch \
    --evaluation_strategy epoch \
    --metric_for_best_model micro_f1 \
    --greater_is_better \
    --max_seq_length $length \
    --segment_length 64 --do_use_stride --do_use_label_wise_attention

#  python3 eval.py \
#  --task_name multilabel \
#  --output_metrics_filepath /checkpoints/metrics/metrics_eval_${output}.json \
#  --model_dir $data_dir/BioGpt/mimic_biogpt_base \
#  --test_filepath /data/50/train.json \
#  --output_dir $output_dir \
#  --max_seq_length $length \
#  --segment_length 64 --do_use_stride --do_use_label_wise_attention

#  rm -r $output_dir
fi
