# nips_submit

This repo code is used both for A100 and 4090.

## data

The data is made public [here1](https://huggingface.co/datasets/Facico/test), [here2](https://huggingface.co/datasets/Facico/test2), [here3](https://huggingface.co/datasets/Facico/test3) before Oct 26th pst


## inference

Total submit for 3 times:
- Dockerfile1  for 1 time
- Dockerfile2  for 1 time
- Dockerfile3  for 1 time

Or you can run the following command:
```
pip install -r requirements.txt
uvicorn qwen_moe_raw:app --port 8000
```

## finetune

Use the [`Dockerfile.finetune`](Dockerfile.finetune) to finetune.

Or you can run the following command:
```
pip install -r requirements-train.txt
bash finetune.sh
```
