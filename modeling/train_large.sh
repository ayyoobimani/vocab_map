WANDB_DISABLED=true python -m torch.distributed.launch --nproc_per_node=8 run.py \
  --model_name_or_path xlm-roberta-large \
  --train_file /mounts/data/proj/ayyoobbig/1000LM/data/1000LM.txt \
  --spm_name /mounts/data/proj/ayyoobbig/1000LM/tokenizer/1000LM_extended_spm.model \
  --output_dir /mounts/data/proj/ayyoobbig/1000LM/LM_large \
  --cache_dir /mounts/data/proj/ayyoobbig/1000LM/cache \
  --per_device_train_batch_size 6 \
  --gradient_accumulation_steps 256 \
  --learning_rate 1e-4 \
  --max_grad_norm 5 \
  --fp16 True \
  --do_train \
  --num_train_epochs 100 \
  --save_steps 25 \
  --logging_steps 10 \
  --ddp_timeout 259200 \

