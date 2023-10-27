from fastapi import FastAPI
import os
import time
import torch
from peft import PeftModel, PeftConfig
from transformers import AutoModelForCausalLM, AutoTokenizer, GenerationConfig
from transformers import BitsAndBytesConfig
import torch
import utils.lora_utils as lora_utils
from utils.prompter import *
from utils.helm_type import *
import utils.debias as debias
from utils.lora_utils import hack_qwen_for_moe
import random
import torch
import numpy as np

os.environ['FALLBACK'] = 'raw'

SEED=int(os.environ.get('SEED',42))
random.seed(SEED)
torch.manual_seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed(SEED)
    torch.cuda.manual_seed_all(SEED)
np.random.seed(SEED)

hack_qwen_for_moe()

MODEL = 'Qwen/Qwen-14B'
LORA=os.environ.get('LORA',None)
DTYPE=torch.bfloat16
MAXLEN = 4096
BIAS=True
use_flash_attention=os.environ.get('FLASH', True)
model = AutoModelForCausalLM.from_pretrained(
    MODEL,
    trust_remote_code=True,
    device_map={"": 0},
    torch_dtype=DTYPE,
    quantization_config=BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=DTYPE,  
        llm_int8_has_fp16_weight=True,
    )
)
print(f'>>> loading {MODEL} finished')
model = lora_utils.add_multi_lora(
    model,
    lora_paths=[
        'lu-vae/qwen-cnn-merged',
        'lu-vae/qwen-mmlu-merged',
        'lu-vae/qwen-truthfulqa-merged',
        'lu-vae/qwen-bbq-merged',
        'lu-vae/qwen-gsm8k-merged',
        'lu-vae/qwen-chat-merged',
    ],
    lora_names=[
        'cnn-dm',
        'mmlu',
        'truthfulqa',
        'bbq',
        'gsm8k',
        'chat',
    ],
)
model.peft_func_map('to_cuda', adapter_names=[
    'cnn-dm',
    'mmlu',
    'truthfulqa',
    'bbq',
    'gsm8k',
    'chat',
],)
model.generation_config = GenerationConfig.from_pretrained(
    MODEL, trust_remote_code=True
)
model = torch.compile(model)
tokenizer = AutoTokenizer.from_pretrained(
    MODEL,
    add_special_tokens=True,
    trust_remote_code=True,
    padding='left',
)
tokenizer.pad_token_id=0

app = FastAPI()

def config_gen(input_data, data_type):
    repetition_penalty = 1.0
    if data_type == 'mmlu':
        input_data.top_k=65
    elif data_type == 'truthfulqa':
        input_data.top_k=105
    elif data_type == 'cnn-dm':
        input_data.top_k=145
    elif "chat" in data_type:
        repetition_penalty = 2.0
    print('gen parameter: ',input_data.top_k,input_data.temperature,repetition_penalty)
    return input_data, repetition_penalty

ACTIVE_LORA_NAME=None

def config_moe(model, data_type):
    global ACTIVE_LORA_NAME
    if data_type == 'cnn-dm':
        if data_type != ACTIVE_LORA_NAME:
            ACTIVE_LORA_NAME=data_type
            model.set_adapter('cnn-dm')
            model.enable_adapter_layers()
    elif data_type == 'mmlu':
        if data_type != ACTIVE_LORA_NAME:
            ACTIVE_LORA_NAME=data_type
            model.set_adapter('mmlu')
            model.enable_adapter_layers()    
    elif data_type == 'truthfulqa':
        if data_type != ACTIVE_LORA_NAME:
            ACTIVE_LORA_NAME=data_type
            model.set_adapter('truthfulqa')
            model.enable_adapter_layers()    
    elif data_type == 'bbq':
        if data_type != ACTIVE_LORA_NAME:
            ACTIVE_LORA_NAME=data_type
            model.set_adapter('bbq')
            model.enable_adapter_layers()  
    elif data_type == 'gsm8k':
        if data_type != ACTIVE_LORA_NAME:
            ACTIVE_LORA_NAME=data_type
            model.set_adapter('gsm8k')
            model.enable_adapter_layers() 
    elif 'chat' in data_type or data_type == 'xsum' or data_type == 'commonsense':
        data_type = 'chat'
        if data_type != ACTIVE_LORA_NAME:
            ACTIVE_LORA_NAME=data_type
            model.set_adapter('chat')
            model.enable_adapter_layers()  
    else:
        data_type = 'raw'
        # back-to-back logic: raw 
        if ACTIVE_LORA_NAME is not None:
            ACTIVE_LORA_NAME=None
            model.disable_adapter_layers()
    # TODO: automatically extract
    print(f'active {model.base_model.model.transformer.h[0].attn.c_attn.active_adapter}')
    print(f'disable {model.base_model.model.transformer.h[0].attn.c_attn.disable_adapters}')

@app.post("/process")
async def process_request(input_data: ProcessRequest) -> ProcessResponse:
    if input_data.seed is not None:
        torch.manual_seed(input_data.seed)

    # data type
    data_type = get_data_type(input_data.prompt)

    # Prompt
    input_data.prompt = config_prompt(input_data.prompt, data_type)

    # Generation parameter
    input_data,repetition_penalty = config_gen(input_data, data_type)
    
    # Lora moe 
    config_moe(model, data_type)

    encoded = tokenizer(input_data.prompt, return_tensors="pt")
    prompt_length = encoded["input_ids"][0].size(0)
    t0 = time.perf_counter()
    encoded = {k: v.to("cuda") for k, v in encoded.items()}
    with torch.no_grad():
        outputs = model.generate(
            **encoded,
            max_new_tokens=input_data.max_new_tokens,
            do_sample=True,
            temperature=input_data.temperature,
            top_k=input_data.top_k,
            return_dict_in_generate=True,
            output_scores=True,
            pad_token_id=0,
            repetition_penalty=repetition_penalty,
        )
    t = time.perf_counter() - t0
    if not input_data.echo_prompt:
        output = tokenizer.decode(outputs.sequences[0][prompt_length:], skip_special_tokens=True)
    else:
        output = tokenizer.decode(outputs.sequences[0], skip_special_tokens=True)
    
    if data_type == 'raw':
        output = dedup(output)
    elif 'chat' in data_type:
        output = output.strip("<|endoftext|>").strip("</s>")
    else:
        output = debias.config_debias(output, data_type)

    print(output)
    tokens_generated = outputs.sequences[0].size(0) - prompt_length
    generated_tokens = []
    log_probs = torch.log(torch.stack(outputs.scores, dim=1).softmax(-1))
    gen_sequences = outputs.sequences[:, encoded["input_ids"].shape[-1]:]
    gen_logprobs = torch.gather(log_probs, 2, gen_sequences[:, :, None]).squeeze(-1)
    top_indices = torch.argmax(log_probs, dim=-1)
    top_logprobs = torch.gather(log_probs, 2, top_indices[:,:,None]).squeeze(-1)
    top_indices = top_indices.tolist()[0]
    top_logprobs = top_logprobs.tolist()[0]
    for t, lp, tlp in zip(gen_sequences.tolist()[0], gen_logprobs.tolist()[0], zip(top_indices, top_logprobs)):
        idx, val = tlp
        tok_str = tokenizer.decode(idx)
        token_tlp = {tok_str: val}
        generated_tokens.append(
            Token(text=tokenizer.decode(t), logprob=lp, top_logprob=token_tlp)
        )
    logprob_sum = gen_logprobs.sum().item()

    return ProcessResponse(
        text=output, tokens=generated_tokens, logprob=logprob_sum, request_time=t
    )

@app.post("/tokenize")
async def tokenize(input_data: TokenizeRequest) -> TokenizeResponse:
    t0 = time.perf_counter()
    encoded = tokenizer(
        input_data.text
    )
    t = time.perf_counter() - t0
    tokens = encoded["input_ids"]
    return TokenizeResponse(tokens=tokens, request_time=t)