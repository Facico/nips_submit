FROM thebloke/ubuntu2204-cuda118-pytorch2
ENV PIP_DEFAULT_TIMEOUT=10000 
COPY ./requirements.txt requirements.txt
RUN DEBIAN_FRONTEND=noninteractive apt-get update && apt-get install -y openssh-client git curl && curl -s https://packagecloud.io/install/repositories/github/git-lfs/script.deb.sh | bash && apt-get install -y git-lfs
RUN pip install --no-cache-dir -r requirements.txt
WORKDIR /workspace
COPY peft peft
RUN cd peft && pip install -e . && cd ..
RUN export GIT_LFS_SKIP_SMUDGE=0
RUN git clone https://lu-vae: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@huggingface.co/lu-vae/qwen-cnn-merged lu-vae/qwen-cnn-merged
RUN git clone https://lu-vae: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@huggingface.co/lu-vae/qwen-mmlu-merged lu-vae/qwen-mmlu-merged
RUN git clone https://lu-vae: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@huggingface.co/lu-vae/qwen-truthfulqa-merged lu-vae/qwen-truthfulqa-merged 
RUN git clone https://lu-vae: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@huggingface.co/lu-vae/qwen-bbq2 lu-vae/qwen-bbq-merged
RUN git clone https://lu-vae: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@huggingface.co/lu-vae/qwen-chat1 lu-vae/qwen-chat-merged
RUN git clone https://lu-vae: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@huggingface.co/lu-vae/qwen-gsm8k-merged lu-vae/qwen-gsm8k-merged
RUN git clone https://huggingface.co/Qwen/Qwen-14B Qwen/Qwen-14B
COPY utils utils
COPY qwen_moe_bbq.py main.py
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "80"]