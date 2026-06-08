llamafactory-cli export \
    --model_name_or_path /home/ps/.cache/modelscope/hub/models/Qwen/Qwen/Qwen3-30B-A3B-Instruct-2507 \
    --adapter_name_or_path /home/ps/hanjiaqi/LlamaFactory/chatjoy/saves/qwen3-30b-moe/chatjoy_v1 \
    --template qwen \
    --finetuning_type lora \
    --export_dir /home/ps/hanjiaqi/LlamaFactory/chatjoy/saves/qwen3-30b-moe/chatjoy_v1_merged \
    --export_size 5 \
    --export_device cpu \
    --export_legacy_format False