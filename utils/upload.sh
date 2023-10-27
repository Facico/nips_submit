# python peft_save_merge.py
huggingface-cli upload --private lu-vae/qwen-bbq-merged /home/lzy/nips/submit/lu-vae/qwen-bbq-merged/adapter_config.json adapter_config.json
huggingface-cli upload --private lu-vae/qwen-bbq-merged /home/lzy/nips/submit/lu-vae/qwen-bbq-merged/adapter_model.bin adapter_model.bin
huggingface-cli upload --private lu-vae/qwen-truthfulqa-merged /home/lzy/nips/submit/lu-vae/qwen-truthfulqa-merged/adapter_config.json adapter_config.json
huggingface-cli upload --private lu-vae/qwen-truthfulqa-merged /home/lzy/nips/submit/lu-vae/qwen-truthfulqa-merged/adapter_model.bin adapter_model.bin

for domain in qwen-mmlu-merged qwen-cnn-merged; do
    huggingface-cli upload --private lu-vae/$domain /home/lzy/nips/submit/lu-vae/$domain/adapter_config.json adapter_config.json
    huggingface-cli upload --private lu-vae/$domain /home/lzy/nips/submit/lu-vae/$domain/adapter_model.bin adapter_model.bin
done