WANDB_DISABLED=true python -m torch.distributed.launch --nproc_per_node=4 run.py \
  --model_name_or_path xlm-roberta-base \
  --train_file /mounts/data/proj/ayyoobbig/1000LM/data/1000LM_small.txt \
  --spm_name /mounts/data/proj/ayyoobbig/1000LM/tokenizer/1000LM_extended_spm.model \
  --output_dir /mounts/data/proj/ayyoobbig/1000LM/LM_small \
  --cache_dir /mounts/data/proj/ayyoobbig/1000LM/cache \
  --per_device_train_batch_size 12 \
  --gradient_accumulation_steps 4 \
  --fp16 True \
  --do_train \
  --num_train_epochs 100 \
  --save_steps 10000 \
  --ddp_timeout 259200 \

