# the finetune code for submission 1

source activate lla

# 1. download opensource dataset
cd finetune
export PYTHONPATH=.

dir=data
git clone https://huggingface.co/datasets/Facico/test $dir
python process_data.py --number 1

# 2. finetuning with lora
yaml_file="qwen.yml"
for file in "$dir"/*; do
    echo $file
    if [[ "$file" =~ .*json ]]; then
        echo $file
        start_time=$(date +%s)

        name=$(basename "$file" .json)
        if [[ "$name" =~ .*chat.* ]]; then
            file="$file;sharegpt"
        fi
        echo "use $file, output in $name" 

        WANDB_NAME=qwen-$name \
        CUDA_VISIBLE_DEVICES=0 \
        accelerate launch -m axolotl.cli.train $yaml_file \
        --datasets $file \
        --output_dir ../outs/qwen-$name \
        --num_epochs 3 \
        --trust_remote_code True 

        end_time=$(date +%s)
        runtime=$((end_time - start_time))
        runtime_minutes=$((runtime / 60))

        echo "$file:  sec: $runtime ;min: $runtime_minutes" >> time_final.txt
    fi
done

# 3. merge some loras (use the parent dir code)
cd ..
export PYTHONPATH=.
CUDA_VISIBLE_DEVICES=0 python utils/peft_save_merge.py
mv outs/qwen-chat_1810 outs/qwen-chat2
mv outs/qwen-cnn_900 outs/qwen-cnn-merged
mv outs/qwen-gsm8k_7473 outs/qwen-gsm8k-merged
mv outs/qwen-mmlu_1129 outs/qwen-mmlu-merged

# 4. You may upload the model to huggingface to run the inference code (Dockerfile1) 
# for domain in qwen-mmlu-merged qwen-cnn-merged qwen-gsm8k-merged qwen-chat2 qwen-bbq-merged qwen-truthfulqa-merged ; do
#     huggingface-cli upload --private lu-vae/"$domain"1 outs/$domain/adapter_config.json adapter_config.json
#     huggingface-cli upload --private lu-vae/"$domain"1 outs/$domain/adapter_model.bin adapter_model.bin
# done