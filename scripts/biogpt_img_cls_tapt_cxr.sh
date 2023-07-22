#!/bin/bash
# normal cpu stuff: allocate cpus, memory
#SBATCH --ntasks=1 --cpus-per-task=10 --mem=12000M
# we run on the gpu partition and we allocate 1 titanx gpus
#SBATCH -p gpu --gres=gpu:titanrtx:1

# shellcheck disable=SC1090
source ~/venv/bin/activate

python3 ~/Thesis/biogpt_seq_classification_tapt.py \
  --task_name=multilabel \
  --seed=52 \
  --evaluation_strategy=epoch \
  --save_strategy=epoch \
  --max_seq_length=1024 \
  --num_train_epochs=2 \
  --per_device_train_batch_size=4 \
  --per_device_eval_batch_size=4 \
  --gradient_accumulation_steps=8 \
  --metric_for_best_model=micro_f1 \
  --should_log=True --compute_perplexity=False --do_use_stride --do_use_label_wise_attention
