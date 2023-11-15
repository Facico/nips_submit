from peft import LoraConfig
from peft.tuners.lora import *
from dataclasses import asdict, dataclass, field, replace
from typing import List, Optional, Tuple, Union
import re
import warnings
import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm
from transformers.pytorch_utils import Conv1D
import enum
import math
from peft.utils import (
    COMMON_LAYERS_PATTERN,
    TRANSFORMERS_MODELS_TO_LORA_TARGET_MODULES_MAPPING,
    ModulesToSaveWrapper,
    _freeze_adapter,
    _get_submodules,
    get_auto_gptq_quant_linear,
    get_quantization_config,
    transpose
)
from peft.import_utils import is_bnb_4bit_available, is_bnb_available
if is_bnb_available():
    import bitsandbytes as bnb

class PeftType(str, enum.Enum):
    PROMPT_TUNING = "PROMPT_TUNING"
    MULTITASK_PROMPT_TUNING = "MULTITASK_PROMPT_TUNING"
    P_TUNING = "P_TUNING"
    PREFIX_TUNING = "PREFIX_TUNING"
    LORA = "LORA"
    ADALORA = "ADALORA"
    ADAPTION_PROMPT = "ADAPTION_PROMPT"
    IA3 = "IA3"
    LLORA = "LLORA"
    
@dataclass
class LLoraConfig(LoraConfig):
    
    small_model_dim: int = field(default=4096, metadata={"help": "Small model dimension"})
    large_model_dim: int = field(default=5120, metadata={"help": "Small model dimension"})
    small_r:int = field(default=32, metadata={"help": "Small model r"})
    large_r:int = field(default=32, metadata={"help": "Large model r"})
    small_model_intermediate_size:int = field(default=11008, metadata={"help": "Small model intermediate size"})
    large_model_intermediate_size:int = field(default=13824, metadata={"help": "Large model intermediate size"})
    
    def __post_init__(self):
        self.peft_type = PeftType.LLORA


class LLoraLayer(LoraLayer):

    def __init__(self, large_in_features: int, large_out_features: int, small_in_features: int, small_out_features: int, **kwargs):
        super().__init__(large_in_features, large_out_features, **kwargs)
        self.small_in_features = small_in_features
        self.small_out_features = small_out_features
        self.large_in_features = large_in_features
        self.large_out_features = large_out_features

    def update_layer(self, adapter_name, small_r, large_r, lora_alpha, lora_dropout, init_lora_weights):
        self.r[adapter_name + "_large"] = large_r
        self.r[adapter_name + "_small"] = small_r
        self.lora_alpha[adapter_name + "_large"] = lora_alpha
        self.lora_alpha[adapter_name + "_small"] = lora_alpha
        if lora_dropout > 0.0:
            lora_dropout_layer = nn.Dropout(p=lora_dropout)
        else:
            lora_dropout_layer = nn.Identity()

        self.lora_dropout.update(nn.ModuleDict({adapter_name: lora_dropout_layer}))
        # Actual trainable parameters
        if small_r > 0 and large_r > 0:
            self.lora_A.update(nn.ModuleDict({adapter_name + "_small": nn.Linear(self.small_in_features, small_r, bias=False),
                                              adapter_name + "_s2l": nn.Linear(small_r, self.large_in_features, bias=False),
                                              adapter_name + "_large": nn.Linear(self.large_in_features, large_r, bias=False)}))
            self.lora_B.update(nn.ModuleDict({adapter_name + "_large": nn.Linear(large_r, self.large_out_features, bias=False),
                                              adapter_name + "_s2l": nn.Linear(self.large_out_features, small_r, bias=False),
                                              adapter_name + "_small": nn.Linear(small_r, self.small_out_features, bias=False)}))
            self.scaling[adapter_name + "_small"] = lora_alpha / small_r
            self.scaling[adapter_name + "_large"] = lora_alpha / large_r
        if init_lora_weights:
            self.reset_lora_parameters(adapter_name)
        
        weight = getattr(self, "weight", None)
        if weight is not None:
            # for linear
            # the layer is already completely initialized, this is an update
            if weight.dtype.is_floating_point or weight.dtype.is_complex:
                self.to(weight.device, dtype=weight.dtype)
            else:
                self.to(weight.device)
        else:
            # for 4bit 8bit
            self.to(self.base_layer.weight.device)
        
        self.set_adapter(self.active_adapters)

    def set_adapter(self, adapter_names):
        """Set the active adapter

        Args:
            adapter_name (str): The name of the adapter to set as active
        """
        if isinstance(adapter_names, str):
            adapter_names = [adapter_names]

        # Deactivate grads on the inactive adapter and activate grads on the active adapter
        for layer_name in self.adapter_layer_names:
            module_dict = getattr(self, layer_name)
            for key, layer in module_dict.items():
                # NOTE!!!
                if key.split('_')[0] in adapter_names:
                    # Note: It is possible that not a single layer is called with requires_grad_(True) here. This may
                    # happen if a completely different adapter layer is being activated.
                    layer.requires_grad_(True)
                else:
                    layer.requires_grad_(False)

        self._active_adapter = adapter_names

    #unuse
    def update_layer_conv2d(self, adapter_name, r, lora_alpha, lora_dropout, init_lora_weights):
        self.r[adapter_name] = r
        self.lora_alpha[adapter_name] = lora_alpha
        if lora_dropout > 0.0:
            lora_dropout_layer = nn.Dropout(p=lora_dropout)
        else:
            lora_dropout_layer = nn.Identity()

        self.lora_dropout.update(nn.ModuleDict({adapter_name: lora_dropout_layer}))
        # Actual trainable parameters
        if r > 0:
            kernel_size = self.kwargs["kernel_size"]
            stride = self.kwargs["stride"]
            padding = self.kwargs["padding"]
            self.lora_A.update(
                nn.ModuleDict({adapter_name: nn.Conv2d(self.in_features, r, kernel_size, stride, padding, bias=False)})
            )
            self.lora_B.update(
                nn.ModuleDict({adapter_name: nn.Conv2d(r, self.out_features, (1, 1), (1, 1), bias=False)})
            )
            self.scaling[adapter_name] = lora_alpha / r
        if init_lora_weights:
            self.reset_lora_parameters(adapter_name)
        self.to(self.weight.device)
    #unuse
    def update_layer_embedding(self, adapter_name, r, lora_alpha, lora_dropout, init_lora_weights):
        self.r[adapter_name] = r
        self.lora_alpha[adapter_name] = lora_alpha
        if lora_dropout > 0.0:
            lora_dropout_layer = nn.Dropout(p=lora_dropout)
        else:
            lora_dropout_layer = nn.Identity()

        self.lora_dropout.update(nn.ModuleDict({adapter_name: lora_dropout_layer}))
        # Actual trainable parameters
        if r > 0:
            weight_A = torch.randn((r, self.in_features), dtype=self.weight.dtype, device=self.weight.device)
            weight_B = torch.randn((self.out_features, r), dtype=self.weight.dtype, device=self.weight.device)
            self.lora_embedding_A.update(nn.ParameterDict({adapter_name: nn.Parameter(weight_A)}))
            self.lora_embedding_B.update(nn.ParameterDict({adapter_name: nn.Parameter(weight_B)}))
            self.scaling[adapter_name] = lora_alpha / r
        if init_lora_weights:
            self.reset_lora_parameters(adapter_name)
        self.to(self.weight.device)

    def reset_lora_parameters(self, adapter_name):
        suffix = ["_small", "_large", "_s2l"]
        for suffix_name in suffix:
            full_name = adapter_name + suffix_name
            if full_name in self.lora_A.keys():
                # initialize A the same way as the default for nn.Linear and B to zero
                nn.init.kaiming_uniform_(self.lora_A[full_name].weight, a=math.sqrt(5))
                # nn.init.zeros_(self.lora_B[full_name].weight)
                nn.init.kaiming_uniform_(self.lora_A[full_name].weight, a=math.sqrt(0.1))
            if full_name in self.lora_embedding_A.keys():
                # initialize a the same way as the default for nn.linear and b to zero
                nn.init.zeros_(self.lora_embedding_A[full_name])
                nn.init.normal_(self.lora_embedding_B[full_name])

class Linear(nn.Linear, LLoraLayer):
    # Lora implemented in a dense layer
    def __init__(
        self,
        adapter_name: str,
        small_in_features: int,
        small_out_features: int,
        large_in_features: int,
        large_out_features: int,
        small_r: int = 0,
        large_r: int = 0,
        lora_alpha: int = 1,
        lora_dropout: float = 0.0,
        fan_in_fan_out: bool = False,  # Set this to True if the layer to replace stores weight like (fan_in, fan_out)
        is_target_conv_1d_layer: bool = False,
        **kwargs,
    ):
        init_lora_weights = kwargs.pop("init_lora_weights", True)
        nn.Linear.__init__(self, small_in_features, small_out_features, **kwargs)
        LLoraLayer.__init__(self, large_in_features=large_in_features, large_out_features=large_out_features, 
                           small_in_features=small_in_features, small_out_features=small_out_features)
        # Freezing the pre-trained weight matrix

        self.fan_in_fan_out = fan_in_fan_out
        if fan_in_fan_out:
            self.weight.data = self.weight.data.T

        nn.Linear.reset_parameters(self)
        self.update_layer(adapter_name, small_r, large_r, lora_alpha, lora_dropout, init_lora_weights)
        self.is_target_conv_1d_layer = is_target_conv_1d_layer
        self.set_adapter(adapter_name)

    def merge(self):
        if self.active_adapter + "_small" not in self.lora_A.keys() and self.active_adapter + "_large" not in self.lora_A.keys():
            return
        if self.merged:
            warnings.warn("Already merged. Nothing to do.")
            return
        if self.r[self.active_adapter + "_small"] > 0 and self.r[self.active_adapter + "_large"] > 0:
            self.weight.data += self.get_delta_weight(self.active_adapter)
            self.merged = True

    def unmerge(self):
        if self.active_adapter + "_small" not in self.lora_A.keys() and self.active_adapter + "_large" not in self.lora_A.keys():
            return
        if not self.merged:
            warnings.warn("Already unmerged. Nothing to do.")
            return
        if self.r[self.active_adapter + "_small"] > 0 and self.r[self.active_adapter + "_large"] > 0:
            self.weight.data -= self.get_delta_weight(self.active_adapter)
            self.merged = False

    def get_delta_weight(self, adapter):
        return (
            transpose(
                self.lora_B[adapter + "_small"].weight @ self.lora_B[adapter + "_s2l"].weight @ \
                self.lora_B[adapter + "_large"].weight @ self.lora_A[adapter + "_large"].weight @ \
                self.lora_B[adapter + "_s2l"].weight @ self.lora_A[adapter + "_small"].weight
                ,self.fan_in_fan_out,
            )
            * self.scaling[adapter + "_large"] #TODO just one?
        )

    def forward(self, x: torch.Tensor):
        previous_dtype = x.dtype
        if self.active_adapter + "_small" not in self.lora_A.keys() and self.active_adapter + "_large" not in self.lora_A.keys():
            return F.linear(x, transpose(self.weight, self.fan_in_fan_out), bias=self.bias)
        if self.disable_adapters:
            if self.r[self.active_adapter + "_small"] > 0 and self.r[self.active_adapter + "_large"] > 0 and self.merged:
                self.unmerge()
            result = F.linear(x, transpose(self.weight, self.fan_in_fan_out), bias=self.bias)
        elif self.r[self.active_adapter + "_small"] > 0 and self.r[self.active_adapter + "_large"] > 0 and not self.merged:
            result = F.linear(x, transpose(self.weight, self.fan_in_fan_out), bias=self.bias)

            x = x.to(self.lora_A[self.active_adapter + "_small"].weight.dtype)

            result += (
                self.lora_B[self.active_adapter + "_small"](
                self.lora_B[self.active_adapter + "_s2l"](
                self.lora_B[self.active_adapter + "_large"](
                    self.lora_A[self.active_adapter + "_large"](
                    self.lora_A[self.active_adapter + "_s2l"](
                    self.lora_A[self.active_adapter + "_small"](self.lora_dropout[self.active_adapter](x))
                    ))
                )))
                * self.scaling[self.active_adapter + "_large"] #TODO
            )
        else:
            result = F.linear(x, transpose(self.weight, self.fan_in_fan_out), bias=self.bias)

        result = result.to(previous_dtype)

        return result

import logging
LOG = logging.getLogger("axolotl")

if is_bnb_available():

    class Linear8bitLt(torch.nn.Module, LLoraLayer):
        # Lora implemented in a dense layer
        def __init__(
            self,
            adapter_name,
            base_layer,
            large_in_features: int,
            large_out_features: int,
            small_r: int = 0,
            large_r: int = 0,
            lora_alpha: int = 1,
            lora_dropout: float = 0.0,
            **kwargs,
        ):
            super().__init__()
            LLoraLayer.__init__(
                self, 
                large_in_features=large_in_features, large_out_features=large_out_features, 
                small_in_features=base_layer.in_features, small_out_features=base_layer.out_features
            )
            self.base_layer = base_layer

            init_lora_weights = kwargs.pop("init_lora_weights", True)
            self.update_layer(adapter_name, small_r, large_r, lora_alpha, lora_dropout, init_lora_weights)
            self.set_adapter(adapter_name)

        def forward(self, x: torch.Tensor, *args, **kwargs):
            if self.disable_adapters:
                if self.merged:
                    self.unmerge()
                result = self.base_layer(x, *args, **kwargs)
            elif self.merged:
                result = self.base_layer(x, *args, **kwargs)
            else:
                result = self.base_layer(x, *args, **kwargs)
                for active_adapter in self.active_adapters:
                    if active_adapter not in self.lora_A.keys():
                        continue

                    requires_conversion = not torch.is_autocast_enabled()
                    if requires_conversion:
                        expected_dtype = result.dtype
                        compute_dtype = self.lora_A[active_adapter + "_small"].weight.dtype
                        if x.dtype != compute_dtype:
                            x = x.to(compute_dtype)
                    
                    output = (
                        self.lora_B[active_adapter + "_small"](
                            self.lora_B[active_adapter + "_s2l"](
                                self.lora_B[active_adapter + "_large"](
                                    self.lora_A[active_adapter + "_large"](
                                        self.lora_A[active_adapter + "_s2l"](
                                            self.lora_A[active_adapter + "_small"](
                                                self.lora_dropout[active_adapter](x)
                                            )
                                        )
                                    )
                                )
                            )
                        )  
                    )

                    if requires_conversion:
                        output = output.to(expected_dtype)
                    
                    output = output * self.scaling[active_adapter + "_large"] #TODO
                    result += output

            return result

    if is_bnb_4bit_available():

        class Linear4bit(torch.nn.Module, LLoraLayer):
            # Lora implemented in a dense layer
            def __init__(
                self,
                adapter_name,
                base_layer,
                large_in_features: int,
                large_out_features: int,
                small_r: int = 0,
                large_r: int = 0,
                lora_alpha: int = 1,
                lora_dropout: float = 0.0,
                **kwargs,
            ):
                super().__init__()
                LLoraLayer.__init__(
                    self, 
                    small_in_features=base_layer.in_features, small_out_features=base_layer.out_features,
                    large_in_features=large_in_features, large_out_features=large_out_features, 
                )

                self.base_layer = base_layer
                init_lora_weights = kwargs.pop("init_lora_weights", True)
                self.update_layer(adapter_name, small_r, large_r, lora_alpha, lora_dropout, init_lora_weights)
                self.set_adapter(adapter_name)

            def forward(self, x: torch.Tensor, *args, **kwargs) -> torch.Tensor:

                if self.disable_adapters:
                    if self.merged:
                        self.unmerge()
                    result = self.base_layer.forward(x, *args, **kwargs)
                elif self.merged:
                    result = self.base_layer.forward(x, *args, **kwargs)
                else:
                    result = self.base_layer.forward(x, *args, **kwargs)
                    # As per Tim Dettmers, for 4bit, we need to defensively clone here.
                    # The reason is that in some cases, an error can occur that backprop
                    # does not work on a manipulated view. This issue may be solved with
                    # newer PyTorch versions but this would need extensive testing to be
                    # sure.
                    result = result.clone()

                    for active_adapter in self.active_adapters:
                        if f'{active_adapter}_small' not in self.lora_A.keys():
                            continue

                        requires_conversion = not torch.is_autocast_enabled()
                        if requires_conversion:
                            expected_dtype = result.dtype
                            x = x.to(self.lora_A[active_adapter + "_small"].weight.dtype)

                        output = (
                            self.lora_B[active_adapter + "_small"](
                                self.lora_B[active_adapter + "_s2l"](
                                    self.lora_B[active_adapter + "_large"](
                                        self.lora_A[active_adapter + "_large"](
                                            self.lora_A[active_adapter + "_s2l"](
                                                self.lora_A[active_adapter + "_small"](
                                                    self.lora_dropout[active_adapter](x)
                                                )
                                            )
                                        )
                                    )
                                )
                            )  
                        )

                        if requires_conversion:
                            output = output.to(expected_dtype)
                        
                        output = output * self.scaling[active_adapter + "_large"] #TODO
                        result += output

                        # LOG.debug(output)

                return result

class QuantLinear(torch.nn.Module, LLoraLayer):
    def __init__(
        self,
        adapter_name,
        quant_linear_module,
        large_in_features: int,
        large_out_features: int,
        small_r: int = 0,
        large_r: int = 0,
        lora_alpha: int = 1,
        lora_dropout: float = 0.0,
        **kwargs,
    ):
        torch.nn.Module.__init__(self)
        LLoraLayer.__init__(
            self, large_in_features=large_in_features, large_out_features=large_out_features,
            small_in_features=quant_linear_module.infeatures, small_out_features=quant_linear_module.outfeatures
        )
        self.quant_linear_module = quant_linear_module
        self.weight = quant_linear_module.qweight
        init_lora_weights = kwargs.pop("init_lora_weights", True)
        self.update_layer(adapter_name, small_r, large_r, lora_alpha, lora_dropout, init_lora_weights)
        self.active_adapter = adapter_name

    def forward(self, x: torch.Tensor):
        result = self.quant_linear_module(x)
        if self.disable_adapters or (self.active_adapter + "_small" not in self.lora_A.keys() and self.active_adapter + "_large" not in self.lora_A.keys()):
            return result
        elif self.r[self.active_adapter + "_small"] > 0 and self.r[self.active_adapter + "_large"] > 0:
            result = result.clone()
            if not torch.is_autocast_enabled():
                expected_dtype = result.dtype
                x = x.to(self.lora_A[self.active_adapter + "_small"].weight.dtype)
                output = (
                    self.lora_B[self.active_adapter + "_small"](
                    self.lora_B[self.active_adapter + "_s2l"](
                    self.lora_B[self.active_adapter + "_large"](
                        self.lora_A[self.active_adapter + "_large"](
                        self.lora_A[self.active_adapter + "_s2l"](
                        self.lora_A[self.active_adapter + "_small"](self.lora_dropout[self.active_adapter](x))
                        ))
                    ))).to(expected_dtype)
                    * self.scaling[self.active_adapter + "_large"] #TODO
                )
            else:
                output = (
                    self.lora_B[self.active_adapter + "_small"](
                    self.lora_B[self.active_adapter + "_s2l"](
                    self.lora_B[self.active_adapter + "_large"](
                        self.lora_A[self.active_adapter + "_large"](
                        self.lora_A[self.active_adapter + "_s2l"](
                        self.lora_A[self.active_adapter + "_small"](self.lora_dropout[self.active_adapter](x))
                        ))
                    )))
                    * self.scaling[self.active_adapter + "_large"] #TODO
                )
            result += output
        return result

    # TODO: Check if it is better as suggested by users https://github.com/PanQiWei/AutoGPTQ/pull/102
    # def reset_lora_parameters(self, adapter_name):
    #     if adapter_name in self.lora_A.keys():
    #         torch.nn.init.xavier_uniform_(self.lora_A[adapter_name].weight)
    #         torch.nn.init.zeros_(self.lora_B[adapter_name].weight)

class LLoraModel(LoraModel):
    def __init__(self, model, config, adapter_name) -> None:
        super().__init__(model, config, adapter_name)
    
    def _create_and_replace(
        self,
        lora_config,
        adapter_name,
        target,
        target_name,
        parent,
        **optionnal_kwargs,
    ):
        bias = hasattr(target, "bias") and target.bias is not None
        kwargs = {
            "small_r": lora_config.small_r,
            "large_r": lora_config.large_r,
            "small_model_dim": lora_config.small_model_dim,
            "large_model_dim": lora_config.large_model_dim,
            "small_model_intermediate_size": lora_config.small_model_intermediate_size,
            "large_model_intermediate_size": lora_config.large_model_intermediate_size,
            "lora_alpha": lora_config.lora_alpha,
            "lora_dropout": lora_config.lora_dropout,
            "fan_in_fan_out": lora_config.fan_in_fan_out,
            "init_lora_weights": lora_config.init_lora_weights,
        }
        kwargs["loaded_in_8bit"] = optionnal_kwargs.pop("loaded_in_8bit", False)
        kwargs["loaded_in_4bit"] = optionnal_kwargs.pop("loaded_in_4bit", False)
        kwargs["bias"] = bias

        quantization_config = get_quantization_config(self.model, method="gptq")
        if quantization_config is not None:
            kwargs["gptq_quantization_config"] = quantization_config

        # TODO: better deal with that
        if isinstance(target, LLoraLayer) and isinstance(target, torch.nn.Conv2d):
            target.update_layer_conv2d( #unuse
                adapter_name,
                lora_config.r,
                lora_config.lora_alpha,
                lora_config.lora_dropout,
                lora_config.init_lora_weights,
            )
        elif isinstance(target, LLoraLayer) and isinstance(target, torch.nn.Embedding):
            target.update_layer_embedding( #unuse
                adapter_name,
                lora_config.small_r,
                lora_config.large_r,
                lora_config.lora_alpha,
                lora_config.lora_dropout,
                lora_config.init_lora_weights,
            )

        elif isinstance(target, LLoraLayer):
            target.update_layer(
                adapter_name,
                lora_config.small_r,
                lora_config.large_r,
                lora_config.lora_alpha,
                lora_config.lora_dropout,
                lora_config.init_lora_weights,
            )
        else:
            new_module = self._create_new_module(lora_config, adapter_name, target, **kwargs)
            # adapter_name + _small == self.active_adapter
            # if adapter_name != self.active_adapter:
            #     # adding an additional adapter: it is not automatically trainable
            #     new_module.requires_grad_(False)
            self._replace_module(parent, target_name, new_module, target)
            
    @staticmethod
    def _create_new_module(lora_config, adapter_name, target, **kwargs):
        gptq_quantization_config = kwargs.get("gptq_quantization_config", None)
        AutoGPTQQuantLinear = get_auto_gptq_quant_linear(gptq_quantization_config)

        loaded_in_8bit = kwargs.pop("loaded_in_8bit", False)
        loaded_in_4bit = kwargs.pop("loaded_in_4bit", False)
        small_model_dim = kwargs.pop("small_model_dim", 4096)
        large_model_dim = kwargs.pop("large_model_dim", 5120)
        small_model_intermediate_size = kwargs.pop("small_model_intermediate_size", 11008)
        large_model_intermediate_size = kwargs.pop("large_model_intermediate_size", 13824)
        large_in_features = large_model_dim
        large_out_features = large_model_dim
        if target.in_features == small_model_intermediate_size:
            large_in_features = large_model_intermediate_size
        if target.out_features == small_model_intermediate_size:
            large_out_features = large_model_intermediate_size
        bias = kwargs.pop("bias", False)

        if loaded_in_8bit and isinstance(target, bnb.nn.Linear8bitLt):
            eightbit_kwargs = kwargs.copy()
            eightbit_kwargs.update(
                {
                    "has_fp16_weights": target.state.has_fp16_weights,
                    "memory_efficient_backward": target.state.memory_efficient_backward,
                    "threshold": target.state.threshold,
                    "index": target.index,
                }
            )
            new_module = Linear8bitLt(
                adapter_name, target,
                large_in_features=large_in_features, large_out_features=large_out_features, 
                bias=bias, **eightbit_kwargs
            )
        elif loaded_in_4bit and is_bnb_4bit_available() and isinstance(target, bnb.nn.Linear4bit):
            fourbit_kwargs = kwargs.copy()
            fourbit_kwargs.update(
                {
                    "compute_dtype": target.compute_dtype,
                    "compress_statistics": target.weight.compress_statistics,
                    "quant_type": target.weight.quant_type,
                }
            )
            new_module = Linear4bit(
                adapter_name, target, 
                large_in_features=large_in_features, large_out_features=large_out_features, 
                bias=bias, **fourbit_kwargs
            )
        elif AutoGPTQQuantLinear is not None and isinstance(target, AutoGPTQQuantLinear):
            new_module = QuantLinear(adapter_name, target, **kwargs)
            target.weight = target.qweight
        elif isinstance(target, torch.nn.Embedding):
            embedding_kwargs = kwargs.copy()
            embedding_kwargs.pop("fan_in_fan_out", None)
            in_features, out_features = target.num_embeddings, target.embedding_dim
            new_module = Embedding(
                adapter_name, in_features, out_features, 
                large_in_features=large_in_features, large_out_features=large_out_features,
                **embedding_kwargs
            )
        elif isinstance(target, torch.nn.Conv2d):
            out_channels, in_channels = target.weight.size()[:2]
            kernel_size = target.weight.size()[2:]
            stride = target.stride
            padding = target.padding
            new_module = Conv2d(adapter_name, in_channels, out_channels, kernel_size, stride, padding,
                    large_in_features=large_in_features, large_out_features=large_out_features, **kwargs)
        else:
            if isinstance(target, torch.nn.Linear):
                in_features, out_features = target.in_features, target.out_features
                if kwargs["fan_in_fan_out"]:
                    warnings.warn(
                        "fan_in_fan_out is set to True but the target module is `torch.nn.Linear`. "
                        "Setting fan_in_fan_out to False."
                    )
                    kwargs["fan_in_fan_out"] = lora_config.fan_in_fan_out = False
            elif isinstance(target, Conv1D):
                in_features, out_features = (
                    target.weight.ds_shape if hasattr(target.weight, "ds_shape") else target.weight.shape
                )
                kwargs["is_target_conv_1d_layer"] = True
                if not kwargs["fan_in_fan_out"]:
                    warnings.warn(
                        "fan_in_fan_out is set to False but the target module is `Conv1D`. "
                        "Setting fan_in_fan_out to True."
                    )
                    kwargs["fan_in_fan_out"] = lora_config.fan_in_fan_out = True
            else:
                raise ValueError(
                    f"Target module {target} is not supported. Currently, only the following modules are supported: "
                    "`torch.nn.Linear`, `torch.nn.Embedding`, `torch.nn.Conv2d`, `transformers.pytorch_utils.Conv1D`."
                )
            new_module = Linear(
                adapter_name, in_features, out_features, 
                large_in_features=large_in_features, large_out_features=large_out_features, 
                bias=bias, **kwargs
            )

        return new_module
    
    def _get_active_adapter(self) -> str:
        active_adapter = None
        for module in self.model.modules():
            if isinstance(module, LLoraLayer):
                active_adapter = module.active_adapter

        if active_adapter is None:
            raise ValueError(
                "Something went wrong, no active adapter could be found, please report the issue on GitHub"
            )
        return active_adapter
    
    def set_adapter(self, adapter_name):
        for module in self.model.modules():
            if isinstance(module, LLoraLayer):
                if module.merged:
                    warnings.warn("Adapter cannot be set when the model is merged. Unmerging the model first.")
                    module.unmerge()
                module.active_adapter = adapter_name
    
    def _unload_and_optionally_merge(self, merge=True, progressbar: bool = False):
        if merge:
            if getattr(self.model, "quantization_method", None) == "gptq":
                raise ValueError("Cannot merge LORA layers when the model is gptq quantized")

        key_list = [key for key, _ in self.model.named_modules() if "lora" not in key]
        desc = "Unloading " + ("and merging " if merge else "") + "model"
        for key in tqdm(key_list, disable=not progressbar, desc=desc):
            try:
                parent, target, target_name = _get_submodules(self.model, key)
            except AttributeError:
                continue
            if isinstance(target, LLoraLayer):
                if isinstance(target, nn.Embedding):
                    new_module = torch.nn.Embedding(target.in_features, target.out_features)
                elif isinstance(target, nn.Conv2d):
                    new_module = torch.nn.Conv2d(
                        target.in_channels,
                        target.out_channels,
                        kernel_size=target.kernel_size,
                        stride=target.stride,
                        padding=target.padding,
                        dilation=target.dilation,
                    )
                elif is_bnb_available() and isinstance(target, bnb.nn.Linear8bitLt):
                    bias = target.bias is not None
                    new_module = bnb.nn.Linear8bitLt(
                        target.in_features,
                        target.out_features,
                        bias=bias,
                        has_fp16_weights=target.state.has_fp16_weights,
                        memory_efficient_backward=target.state.memory_efficient_backward,
                        threshold=target.state.threshold,
                        index=target.index,
                        device=target.weight.device,
                    )
                elif is_bnb_4bit_available() and isinstance(target, bnb.nn.Linear4bit):
                    bias = target.bias is not None
                    new_module = bnb.nn.Linear4bit(
                        target.in_features,
                        target.out_features,
                        bias=bias,
                        compute_dtype=target.compute_dtype,
                        compress_statistics=target.weight.compress_statistics,
                        quant_type=target.weight.quant_type,
                        device=target.weight.device,
                    )
                else:
                    bias = target.bias is not None
                    if getattr(target, "is_target_conv_1d_layer", False):
                        new_module = Conv1D(target.out_features, target.in_features)
                    else:
                        new_module = torch.nn.Linear(target.in_features, target.out_features, bias=bias)
                if merge:
                    target.merge()
                self._replace_module(parent, target_name, new_module, target)

            # save any additional trainable modules part of `modules_to_save`
            if isinstance(target, ModulesToSaveWrapper):
                setattr(parent, target_name, target.modules_to_save[target.active_adapter])

        return self.model
    
    #unfinished
    #add_weighted_adapter, _svd_weighted_adapter