HOST_NUM=1
HOST_GPU_NUM=1
TJ_INSTANCE_ID="llama13b"
CHIEF_IP="localhost:29400"

torchrun \
    --nnodes=$HOST_NUM \
    --nproc_per_node=$HOST_GPU_NUM \
    --rdzv_id=$TJ_INSTANCE_ID \
    --rdzv_backend=c10d \
    --rdzv_endpoint=$CHIEF_IP \
    --master_port=12345 \
main.py \
    --model_name_or_path llama_13B_hf \
    --data_path ./alpaca_data.json \
    --bf16 True \
    --output_dir llama_13B_output \
    --num_train_epochs 3 \
    --per_device_train_batch_size 4 \
    --per_device_eval_batch_size 4 \
    --evaluation_strategy "no" \
    --save_strategy "steps" \
    --save_steps 2000 \
    --save_total_limit 1 \
    --learning_rate 2e-5 \
    --weight_decay 0. \
    --warmup_ratio 0.03 \
    --lr_scheduler_type "cosine" \
    --logging_steps 1 \
    --tf32 True \
    --deepspeed ZeRO3.json
#--gradient_accumulation_steps 8 \
#    --fsdp "full_shard auto_wrap" \
    # --fsdp_transformer_layer_cls_to_wrap 'LlamaDecoderLayer' \