HF_DATASETS_OFFLINE=1
TRANSFORMERS_OFFLINE=1
DIFFUSERS_OFFLINE=1
HF_ENDPOINT=https://hf-mirror.com

torchrun --nproc_per_node 4 --standalone train_dreambooth_colossalai.py \
  --pretrained_model_name_or_path="CompVis/stable-diffusion-v1-4"  \
  --instance_data_dir="data" \
  --output_dir="output" \
  --class_data_dir="class" \
  --instance_prompt="a photo of qwertyuiop cat" \
  --class_prompt="a photo of cat" \
  --with_prior_preservation \
  --prior_loss_weight=1.0 \
  --resolution=512 \
  --plugin="gemini" \
  --train_batch_size=1 \
  --learning_rate=5e-6 \
  --lr_scheduler="constant" \
  --lr_warmup_steps=0 \
  --max_train_steps=1000

