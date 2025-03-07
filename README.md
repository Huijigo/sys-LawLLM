# sys-LawLLM

For the training of the generative model, we use the alignment-handbook framework to perform QLora fine-tuning on the model. You need to navigate to the *GenerationModel/alignment-handbook* file directory and run the following commands. 
```bash
ACCELERATE_LOG_LEVEL=info accelerate launch --config_file recipes/accelerate_configs/multi_gpu.yaml --num_processes=1 scripts/run_sft.py recipes/llama3-8b/sft/config_qlora.yaml --load_in_4bit=true
```

At the same time, you need to set the data source and the model saving path in *recipes/llama3-8b/sft/config_qlora.yaml*.

For the training of the retrieval model, navigate to the */RetrievalModel* directory and use the following commands for training.

```bash
torchrun --nproc_per_node 1 \
    -m FlagEmbedding.finetune.embedder.encoder_only.base \
    --model_name_or_path BAAI/bge-large-zh-v1.5 \
    --cache_dir ./cache/model \
    --train_data  ./sys-LawLLM/RetrievalModel/P2_RAGModel/version_data_5 \
    --cache_path ./cache/data \
    --train_group_size 10 \
    --query_max_len 512 \
    --passage_max_len 512 \
    --pad_to_multiple_of 8 \
    --knowledge_distillation True \
    --output_dir ./model \
    --overwrite_output_dir \
    --learning_rate 1e-5 \
    --num_train_epochs 2 \
    --per_device_train_batch_size 32 \
    --dataloader_drop_last True \
    --warmup_ratio 0.1 \
    --gradient_checkpointing \
    --deepspeed ./ds_stage0.json \
    --logging_steps 1 \
    --save_steps 200 \
    --negatives_cross_device \
    --temperature 0.02 \
    --sentence_pooling_method cls \
    --normalize_embeddings True \
    --kd_loss_type kl_div 
```