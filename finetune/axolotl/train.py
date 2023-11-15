"""Prepare and train a model on a dataset. Can also infer from a model or merge lora"""

import logging
import os
import signal
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import torch
import transformers
from datasets import Dataset
from optimum.bettertransformer import BetterTransformer

from axolotl.common.cli import TrainerCliArgs
import axolotl.pdb_extension
from axolotl.logging_config import configure_logging
from axolotl.utils.dict import DictDefault
from axolotl.utils.models import load_model, load_tokenizer
from axolotl.utils.trainer import setup_trainer

project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
src_dir = os.path.join(project_root, "src")
sys.path.insert(0, src_dir)

configure_logging()
LOG = logging.getLogger("axolotl.train")


@dataclass
class TrainDatasetMeta:
    """
    dataclass to capture the dataset specific options for training
    """

    train_dataset: Dataset
    eval_dataset: Optional[Dataset] = None
    total_num_steps: Optional[int] = None


max_outliner = 0
import warnings
def debug_nan(model):

    class nan_hook:
        def __init__(self,name, module):
            # module.__class__.__name__ 无法给出层数信息
            self.name=name
            module.register_forward_hook(self._hook)

        def _hook(self, module, inp, output):
            # 带lora的时候不准
            # printf(self.name)
            
            if not isinstance(output, tuple):
                outputs = [output]
            else:
                outputs = output

            for i, out in enumerate(outputs):

                if out is None:
                    continue
                if self.name == 'model':
                    # dataclass
                    continue
                if isinstance(out, dict):
                    # for k,v in out.__dict__.items():
                    #     try:
                    #         print(k, v.max())
                    #     except:
                    #         pass
                    return
                # else:
                #     printf(out.max())
                
                nan_mask = torch.isnan(out)
                if nan_mask.any():
                    raise RuntimeError(f"Found NAN in {self.name} output {i} at indices: ", nan_mask.nonzero())
                inf_mask = torch.isinf(out)
                if inf_mask.any():
                    raise RuntimeError(f"Found INF in {self.name} output {i} at indices: ", inf_mask.nonzero())
                outliner = out.abs().max()
                if outliner > 1000:
                    # raise RuntimeError(f"Found outlier in {self.name} output {out_max}: ", out.argmax())
                    # warnings.warn(f"Found outlier in {self.name} output {out_max}: {out.argmax()}" )
                    global max_outliner
                    max_outliner = max(max_outliner, outliner.item())

            # torch.isinf(hidden_states).any()
            # torch.isinf(hidden_states).nonzero()
    
    # for submodule in model.modules():
    for name,submodule in model.named_modules():
        nan_hook(name, submodule)

def compute_loss(self, model, inputs, return_outputs=False):
    """
    How the loss is computed by Trainer. By default, all models return the loss in the first element.

    Subclass and override for custom behavior.
    """
    if self.label_smoother is not None and "labels" in inputs:
        labels = inputs.pop("labels")
    else:
        labels = None
    outputs = model(**inputs)

    if self.cfg.debug_gen:
        # shift_logits = logits[..., :-1, :].contiguous()
        # shift_labels = labels[..., 1:].contiguous()
        idxs = torch.where(inputs['labels'][0]!=-100)[0]
        in_text = self.tokenizer.decode(inputs['labels'][0][idxs])
        out_text = self.tokenizer.decode(torch.argmax(outputs['logits'],dim=-1)[0][idxs-1])
        LOG.debug({"in": in_text[:100]})
        LOG.debug({"out": out_text[:100]})

    if self.cfg.debug_nan and 'llama' in self.cfg.base_model.lower():
        if int(os.getenv('WORLD_SIZE',1)) > 1:
            _model = model.module
        else:
            _model = model
        LOG.debug(f"A : {_model.base_model.model.model.layers[0].self_attn.q_proj.lora_A['default_small'].weight.abs().max()}")
        LOG.debug(_model.base_model.model.model.layers[0].self_attn.q_proj.lora_A['default_small'].weight)
        LOG.debug(f"B : {_model.base_model.model.model.layers[0].self_attn.q_proj.lora_B['default_small'].weight.abs().max()}")
        LOG.debug(_model.base_model.model.model.layers[0].self_attn.q_proj.lora_B['default_small'].weight)
    
    if self.cfg.debug_nan and torch.isnan(outputs.loss):
        raise Exception('loss is nan')

    if self.args.past_index >= 0:
        self._past = outputs[self.args.past_index]
    
    if labels is not None:
        loss = self.label_smoother(outputs, labels, shift_labels=True)
    else:
        if isinstance(outputs, dict) and "loss" not in outputs:
            raise ValueError(
                "The model did not return a loss from the inputs, only the following keys: "
                f"{','.join(outputs.keys())}. For reference, the inputs it received are {','.join(inputs.keys())}."
            )
        # We don't use .loss here since the model may return tuples instead of ModelOutput.
        loss = outputs["loss"] if isinstance(outputs, dict) else outputs[0]

    return (loss, outputs) if return_outputs else loss

class CustomCallback(transformers.TrainerCallback):

    def __init__(self, trainer):
        self.logger = logging.getLogger("transformers.trainer")
        self.trainer = trainer

    def on_train_begin(self, args, state, control, **kwargs):
        if self.trainer.cfg.save_before_train:
            self.trainer._save_checkpoint(self.trainer.model, trial=None)

    def on_log(self, args, state, control, logs, **kwargs):
        self.logger.info(logs)

def train(
    *,
    cfg: DictDefault,
    cli_args: TrainerCliArgs,
    dataset_meta: TrainDatasetMeta,
):
    # load the tokenizer first
    LOG.info(f"loading tokenizer... {cfg.tokenizer_config or cfg.base_model_config}")
    tokenizer = load_tokenizer(cfg)

    train_dataset = dataset_meta.train_dataset
    eval_dataset = dataset_meta.eval_dataset
    total_num_steps = dataset_meta.total_num_steps

    # Load the model and tokenizer
    LOG.info("loading model and (optionally) peft_config...")
    model, peft_config = load_model(cfg, tokenizer, inference=cli_args.inference)

    safe_serialization = cfg.save_safetensors is True

    if cfg.resume_from_checkpoint is None and cfg.auto_resume_from_checkpoints:
        possible_checkpoints = [
            str(cp) for cp in Path(cfg.output_dir).glob("checkpoint-*")
        ]
        if len(possible_checkpoints) > 0:
            sorted_paths = sorted(
                possible_checkpoints,
                key=lambda path: int(path.split("-")[-1]),
            )
            cfg.resume_from_checkpoint = sorted_paths[-1]
            LOG.info(
                f"Using Auto-resume functionality to start with checkpoint at {cfg.resume_from_checkpoint}"
            )
    resume_from_checkpoint = cfg.resume_from_checkpoint

    trainer = setup_trainer(
        cfg, train_dataset, eval_dataset, model, tokenizer, total_num_steps
    )

    model.config.use_cache = False

    # go ahead and presave, so we have the adapter config available to inspect
    if peft_config:
        LOG.info(f"Pre-saving adapter config to {cfg.output_dir}")
        peft_config.save_pretrained(cfg.output_dir)
    # additionally presave the tokenizer and model configs
    if not Path(cfg.output_dir).is_dir():
        os.makedirs(cfg.output_dir, exist_ok=True)
    tokenizer.save_pretrained(str(Path(cfg.output_dir)))
    model.config.save_pretrained(str(Path(cfg.output_dir)))

    # In case we want to stop early with ctrl+c, this is a nice to have to save the pretrained model
    if cfg.local_rank == 0:

        # def terminate_handler(_, __, model):
        #     if cfg.flash_optimum:
        #         model = BetterTransformer.reverse(model)
        #     model.save_pretrained(cfg.output_dir, safe_serialization=safe_serialization)
        #     sys.exit(0)

        # signal.signal(
        #     signal.SIGINT, lambda signum, frame: terminate_handler(signum, frame, model)
        # )
        pass

    badge_markdown = """[<img src="https://raw.githubusercontent.com/OpenAccess-AI-Collective/axolotl/main/image/axolotl-badge-web.png" alt="Built with Axolotl" width="200" height="32"/>](https://github.com/OpenAccess-AI-Collective/axolotl)"""
    transformers.modelcard.AUTOGENERATED_TRAINER_COMMENT += f"\n{badge_markdown}"

    LOG.info("Starting trainer...")
    if cfg.group_by_length:
        LOG.info("hang tight... sorting dataset for group_by_length")

    setattr(trainer.__class__, 'compute_loss', compute_loss)
    trainer.tokenizer = tokenizer
    trainer.cfg = cfg
    trainer.add_callback(CustomCallback(trainer))
    if cfg.debug_nan:
        debug_nan(model)

    pretrain_hooks(cfg, trainer)
    if cfg.flash_optimum:
        with torch.backends.cuda.sdp_kernel(
            enable_flash=True, enable_math=True, enable_mem_efficient=True
        ):
            trainer.train(resume_from_checkpoint=resume_from_checkpoint)
    else:
        trainer.train(resume_from_checkpoint=resume_from_checkpoint)
    post_train_hooks(cfg, trainer)

    LOG.info(f"Training Completed!!! Saving pre-trained model to {cfg.output_dir}")

    if trainer.is_fsdp_enabled:
        trainer.accelerator.state.fsdp_plugin.set_state_dict_type("FULL_STATE_DICT")
        LOG.info("Set FSDP state dict type to FULL_STATE_DICT for saving.")

    if cfg.relora_steps:
        if cfg.adapter == "lora" and not (cfg.load_in_4bit or cfg.load_in_8bit):
            model = model.merge_and_unload()
        else:
            # final model weights have already been saved by `ReLoRACallback.on_train_end`
            return model, tokenizer

    # TODO do we need this fix? https://huggingface.co/docs/accelerate/usage_guides/fsdp#saving-and-loading
    # only save on rank 0, otherwise it corrupts output on multi-GPU when multiple processes attempt to write the same file
    if cfg.fsdp:
        trainer.save_model(cfg.output_dir)
    elif cfg.local_rank == 0:
        if cfg.flash_optimum:
            model = BetterTransformer.reverse(model)

        model.save_pretrained(cfg.output_dir, safe_serialization=safe_serialization)

    if not cfg.hub_model_id:
        trainer.create_model_card(model_name=cfg.output_dir.lstrip("./"))

    return model, tokenizer

def pretrain_hooks(cfg, trainer):
    if cfg.noisy_embedding_alpha:
        from axolotl.monkeypatch import neft_embeddings
        neft_embeddings.pretrain_hook(cfg, trainer)


def post_train_hooks(cfg, trainer):
    if cfg.noisy_embedding_alpha:
        from axolotl.monkeypatch import neft_embeddings
        neft_embeddings.post_train_hook(cfg, trainer)