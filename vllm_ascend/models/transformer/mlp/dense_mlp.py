from typing import Optional

import torch

from vllm.distributed import tensor_model_parallel_all_reduce
from vllm.model_executor.layers.quantization import QuantizationConfig
from vllm.model_executor.layers.activation import SiluAndMul
from vllm.model_executor.layers.linear import MergedColumnParallelLinear, RowParallelLinear, ReplicatedLinear, UnquantizedLinearMethod


from vllm_ascend.quantization.quant_config import AscendLinearMethod
from vllm_ascend.quantization.w8a8_dynamic import AscendW8A8DynamicLinearMethod


class DenseMLP(torch.nn.Module):

    def __init__(
        self,
        hidden_size: int,
        intermediate_size: int,
        hidden_act: str,
        quant_config: Optional[QuantizationConfig] = None,
        reduce_results: bool = True,
        force_replicate: bool = False,
        prefix: str = "",
    ) -> None:
        super().__init__()
        self.reduce_results = reduce_results
        self.force_replicate = force_replicate
        if not force_replicate:
            self.gate_up_proj = MergedColumnParallelLinear(
                hidden_size, [intermediate_size] * 2,
                bias=False,
                quant_config=quant_config,
                prefix=f"{prefix}.gate_up_proj")
            self.down_proj = RowParallelLinear(intermediate_size,
                                               hidden_size,
                                               bias=False,
                                               quant_config=quant_config,
                                               reduce_results=False,  # will reduce output manual
                                               prefix=f"{prefix}.down_proj")
        else:
            self.gate_up_proj = ReplicatedLinear(
                hidden_size, [intermediate_size] * 2,
                bias=False,
                quant_config=quant_config,
                prefix=f"{prefix}.gate_up_proj")
            self.down_proj = ReplicatedLinear(intermediate_size,
                                              hidden_size,
                                              bias=False,
                                              quant_config=quant_config,
                                              prefix=f"{prefix}.down_proj")
        if hidden_act != "silu":
            raise ValueError(f"Unsupported activation: {hidden_act}. "
                             "Only silu is supported for now.")

        quant_method = self.gate_up_proj.quant_method
        if isinstance(quant_method, UnquantizedLinearMethod):
            self.act_fn = SiluAndMul()
        elif (isinstance(quant_method, AscendLinearMethod) and isinstance(
                quant_method.quant_method, AscendW8A8DynamicLinearMethod)):
            # TODO(sdmyzlp): Currently preserved as before:
            # 1. The only quantization supported for silu is W8A8Dynamic
            # 2. Output dtype of gate_up/down is fixed to be int32/bfloat16
            #
            # Maybe one can implement a better and more general configuration
            # scheme, e.g. by somehow passing around the tweaked `quant_config`
            self.act_fn = SiluAndMul(
                # Use lazy binding, for `weight_scale_fp32` is accessible
                # only after `process_weights_after_loading`.
                weight_scale=lambda: self.gate_up_proj.weight_scale_fp32)
            # To be consumed by AscendW8A8DynamicLinearMethod.apply()
            self.gate_up_proj._ascend_quant_config = {
                "output_dtype": torch.int32,
                "pertoken_scale": False,
                "return_scale": True,
            }
            self.down_proj._ascend_quant_config = {
                "output_dtype": torch.bfloat16,
                "pertoken_scale": True,
                "return_scale": False,
            }
        else:
            raise NotImplementedError(
                f"Quantization with [{type(quant_method)}] is NOT supported")

    def forward(self, x):
        gate_up, _ = self._forward_comp_gate_up(x)
        x = self._forward_comp_act_fn(gate_up)
        x, _ = self._forward_comp_down(x)
        x = self._forward_comm_down(x)
        return x

    def forward_generator(self, x):
        gate_up, _ = self._forward_comp_gate_up(x)
        x = self._forward_comp_act_fn(gate_up)
        x, _ = self._forward_comp_down(x)
        x = self._forward_comm_down(x)
        yield x

    def _forward_comp_gate_up(self, x):
        return self.gate_up_proj(x)

    def _forward_comp_act_fn(self, x):
        return self.act_fn(x)

    def _forward_comp_down(self, x):
        return self.down_proj(x)

    def _forward_comm_down(self, x):
        if not self.force_replicate and self.reduce_results and self.down_proj.tp_size > 1:
            return tensor_model_parallel_all_reduce(x)
        return x
