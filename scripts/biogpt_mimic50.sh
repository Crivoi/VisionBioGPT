#!/bin/bash

module load intel-mkl/2020.3.304
module load python3/3.9.2
module load cuda/11.2.2
module load cudnn/8.1.1-cuda11
module load openmpi/4.1.0
module load magma/2.6.0
module load fftw3/3.3.8
module load pytorch/1.9.0

seed=52
max_seq_length=1024
epochs=3

python3 train.py \
  --task_name=multilabel \
  --seed=$seed \
  --evaluation_strategy=epoch \
  --save_strategy=epoch \
  --max_seq_length=$max_seq_length \
  --num_train_epochs=$epochs \
  --per_device_train_batch_size=8 \
  --per_device_eval_batch_size=2 \
  --metric_for_best_model=micro_f1 \
  --should_log=True \
  --segment_length=64 --do_use_stride --do_use_label_wise_attention
