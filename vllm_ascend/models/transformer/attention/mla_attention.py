# Copyright Huawei Technologies Co., Ltd. 2025. All rights reserved.
# Copyright 2025 The vLLM team.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# Extracted according to DeepseekV2MLAAttention from vllm/model_executor/models/deepseek_v2.py
# This file is a part of the vllm-ascend project.

from typing import Any, Optional

import torch
from torch import nn
from transformers import PretrainedConfig

from vllm.attention import Attention
from vllm.config import CacheConfig
from vllm.distributed import (get_tensor_model_parallel_world_size,
                              tensor_model_parallel_all_reduce)
from vllm.model_executor.layers.layernorm import RMSNorm
from vllm.model_executor.layers.linear import (ColumnParallelLinear,
                                               ReplicatedLinear,
                                               RowParallelLinear)
from vllm.model_executor.layers.quantization import QuantizationConfig
from vllm.model_executor.layers.rotary_embedding import get_rope

from vllm_ascend.utils import npu_prefetch


def yarn_get_mscale(scale: float = 1, mscale: float = 1) -> float:
    import math
    if scale <= 1:
        return 1.0
    return 0.1 * mscale * math.log(scale) + 1.0


class MLAAttention(nn.Module):

    def __init__(
            self,
            config: PretrainedConfig,
            hidden_size: int,
            num_heads: int,
            qk_nope_head_dim: int,
            qk_rope_head_dim: int,
            v_head_dim: int,
            q_lora_rank: Optional[int],
            kv_lora_rank: int,
            rope_theta: float = 10000,
            rope_scaling: Optional[dict[str, Any]] = None,
            max_position_embeddings: int = 8192,
            cache_config: Optional[CacheConfig] = None,
            quant_config: Optional[QuantizationConfig] = None,
            prefix: str = "",
    ) -> None:
        super().__init__()
        self.hidden_size = hidden_size
        self.qk_nope_head_dim = qk_nope_head_dim
        self.qk_rope_head_dim = qk_rope_head_dim
        self.qk_head_dim = qk_nope_head_dim + qk_rope_head_dim
        self.v_head_dim = v_head_dim

        self.q_lora_rank = q_lora_rank
        self.kv_lora_rank = kv_lora_rank

        self.num_heads = num_heads
        self.tp_size = get_tensor_model_parallel_world_size()
        assert num_heads % self.tp_size == 0
        self.num_local_heads = num_heads // self.tp_size

        self.layers = config.num_hidden_layers
        self.first_k_dense_replace = config.first_k_dense_replace

        self.scaling = self.qk_head_dim ** -0.5
        self.rope_theta = rope_theta
        self.max_position_embeddings = max_position_embeddings

        self.prefix = prefix
        self.debug_layer_idx = int(self.prefix.split(".")[-2])

        if self.q_lora_rank is not None:
            self.q_a_proj = ReplicatedLinear(self.hidden_size,
                                             self.q_lora_rank,
                                             bias=False,
                                             quant_config=quant_config,
                                             prefix=f"{prefix}.q_a_proj")
            self.q_a_layernorm = RMSNorm(self.q_lora_rank,
                                         eps=config.rms_norm_eps)
            self.q_b_proj = ColumnParallelLinear(q_lora_rank,
                                                 self.num_heads *
                                                 self.qk_head_dim,
                                                 bias=False,
                                                 quant_config=quant_config,
                                                 prefix=f"{prefix}.q_b_proj")
        else:
            self.q_proj = ColumnParallelLinear(self.hidden_size,
                                               self.num_heads *
                                               self.qk_head_dim,
                                               bias=False,
                                               quant_config=quant_config,
                                               prefix=f"{prefix}.q_proj")

        self.kv_a_proj_with_mqa = ReplicatedLinear(
            self.hidden_size,
            self.kv_lora_rank + self.qk_rope_head_dim,
            bias=False,
            quant_config=quant_config,
            prefix=f"{prefix}.kv_a_proj_with_mqa")
        self.kv_a_layernorm = RMSNorm(self.kv_lora_rank,
                                      eps=config.rms_norm_eps)
        self.kv_b_proj = ColumnParallelLinear(
            self.kv_lora_rank,
            self.num_heads * (self.qk_nope_head_dim + self.v_head_dim),
            bias=False,
            quant_config=quant_config,
            prefix=f"{prefix}.kv_b_proj")

        # TODO: if o is calculated in Attention:
        #  1、[TO FIX] three params needed in forward func, now forward func of RowParallelLinear need only one.
        #  2、unable to parse comp and comm in mla_attention forward
        self.o_proj = RowParallelLinear(self.num_heads * self.v_head_dim,
                                        self.hidden_size,
                                        bias=False,
                                        # reduce_results=False,            # if comm need to split out
                                        quant_config=quant_config,
                                        prefix=f"{prefix}.o_proj")

        if rope_scaling:
            rope_scaling["rope_type"] = 'deepseek_yarn'
        self.rotary_emb = get_rope(qk_rope_head_dim,
                                   rotary_dim=qk_rope_head_dim,
                                   max_position=max_position_embeddings,
                                   base=rope_theta,
                                   rope_scaling=rope_scaling,
                                   is_neox_style=False)
        if rope_scaling:
            mscale_all_dim = rope_scaling.get("mscale_all_dim", False)
            scaling_factor = rope_scaling["factor"]
            mscale = yarn_get_mscale(scaling_factor, float(mscale_all_dim))
            self.scaling = self.scaling * mscale * mscale

        # In the MLA backend, kv_cache includes both k_c and
        # pe (i.e. decoupled position embeddings). In particular,
        # the concat_and_cache_mla op requires
        #     k_c.size(1) + k_pe.size(1) == kv_cache.size(2)
        # i.e.
        #     kv_lora_rank + qk_rope_head_dim == head_size
        self.mla_attn = Attention(
            num_heads=self.num_local_heads,
            head_size=self.kv_lora_rank + self.qk_rope_head_dim,
            scale=self.scaling,
            num_kv_heads=1,
            cache_config=cache_config,
            quant_config=quant_config,
            prefix=f"{prefix}.attn",
            use_mla=True,
            # MLA Args
            q_lora_rank=self.q_lora_rank,
            kv_lora_rank=self.kv_lora_rank,
            qk_nope_head_dim=self.qk_nope_head_dim,
            qk_rope_head_dim=self.qk_rope_head_dim,
            qk_head_dim=self.qk_head_dim,
            v_head_dim=self.v_head_dim,
            rotary_emb=self.rotary_emb,
            q_proj=self.q_proj if self.q_lora_rank is None else self.q_b_proj,
            kv_a_proj_with_mqa=self.kv_a_proj_with_mqa,
            kv_a_layernorm=self.kv_a_layernorm,
            kv_b_proj=self.kv_b_proj,
            o_proj=self.o_proj,
        )

    def forward(
        self,
        positions: torch.Tensor,
        hidden_states: torch.Tensor,
    ) -> torch.Tensor:
        # TODO: q calculation is invoked in Attention
        hidden_states_or_q_c, kv_c_normed, k_pe = self._froward_comp_qkv(hidden_states)
        # TODO: rope calculation is invoked in Attention
        # q, k_pe = self._forward_comp_rope(positions, q, k_pe)
        # TODO: q calculation is invoked in Attention
        attn_out = self._forward_comp_attn(hidden_states, hidden_states_or_q_c, kv_c_normed, k_pe)
        # TODO: o calculation is invoked in Attention(vllm-ascend)
        # output = self._forward_comp_output(attn_out)
        # output = self._forward_comm_output(output)
        return attn_out

    def forward_generator(
        self,
        positions: torch.Tensor,
        hidden_states: torch.Tensor,
    ) -> torch.Tensor:
        # q calculation is invoked in Attention
        hidden_states_or_q_c, kv_c_normed, k_pe = self._froward_comp_qkv(hidden_states)
        # rope calculation is invoked in Attention
        # q, k_pe = self._forward_comp_rope(positions, q, k_pe)
        # q calculation is invoked in Attention
        attn_out = self._forward_comp_attn(hidden_states, hidden_states_or_q_c, kv_c_normed, k_pe)
        # o calculation is invoked in Attention(vllm-ascend)
        # output = self._forward_comp_output(attn_out)
        # output = self._forward_comm_output(output)
        yield attn_out

    def _froward_comp_qkv(
        self,
        hidden_states: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        if self.q_lora_rank is not None:
            npu_prefetch(self.q_a_proj.weight, hidden_states, enabled=False)
            q_c = self.q_a_proj(hidden_states)[0]
            hidden_states_or_q_c = self.q_a_layernorm(q_c)
            # TODO: q calculation is invoked in Attention
            # q = self.q_b_proj(q_c)[0]
        else:
            hidden_states_or_q_c = hidden_states
            # q = self.q_proj(hidden_states)[0]
        kv_c, k_pe = self.kv_a_proj_with_mqa(hidden_states)[0].split(
            [self.kv_lora_rank, self.qk_rope_head_dim], dim=-1)
        kv_c_normed = self.kv_a_layernorm(kv_c.contiguous())
        # TODO: q calculation is invoked in Attention
        # q = q.view(-1, self.num_local_heads, self.qk_head_dim)
        # Add head dim of 1 to k_pe
        k_pe = k_pe.unsqueeze(1)
        return hidden_states_or_q_c, kv_c_normed, k_pe

    def _forward_comp_rope(
        self,
        positions: torch.Tensor,
        q: torch.Tensor,
        k_pe: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        q[..., self.qk_nope_head_dim:], k_pe = self.rotary_emb(
            positions, q[..., self.qk_nope_head_dim:], k_pe)
        return q, k_pe

    def _forward_comp_attn(
        self,
        hidden_states: torch.Tensor,
        hidden_states_or_q_c: torch.Tensor,
        kv_c_normed: torch.Tensor,
        k_pe: torch.Tensor
    ) -> torch.Tensor:
        """
        refer to vllm_ascend CustomDeepseekV2MLAAttention in deepseek_v2.py
        """
        if self.debug_layer_idx < self.first_k_dense_replace:
            output_shape = hidden_states.shape
        else:
            num_tokens = hidden_states_or_q_c.shape[0]
            rows = num_tokens // self.tp_size
            if num_tokens % self.tp_size:
                rows += 1
            output_shape = (rows, hidden_states.shape[1])
        return self.mla_attn(hidden_states_or_q_c, kv_c_normed, k_pe, out_shape=output_shape)

    def _forward_comp_output(
        self,
        attn_out: torch.Tensor
    ) -> torch.Tensor:
        return self.o_proj(attn_out)[0]

    def _forward_comm_output(
        self,
        output_parallel: torch.Tensor
    ) -> torch.Tensor:
        output = output_parallel
        if self.tp_size > 1:
            output = tensor_model_parallel_all_reduce(output_parallel)
        return output
