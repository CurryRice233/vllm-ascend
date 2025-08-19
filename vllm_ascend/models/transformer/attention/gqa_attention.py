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
# Extracted according to Qwen3MoeAttention from vllm/model_executor/models/qwen3_moe.py
# and Llama4Attention from vllm/model_executor/models/llama4.py
# This file is a part of the vllm-ascend project.

from typing import Optional, Any

import torch
from torch import nn
from transformers import PretrainedConfig

from vllm.attention import Attention
from vllm.config import CacheConfig, CompilationLevel, VllmConfig
from vllm.distributed import (get_pp_group,
                              get_tensor_model_parallel_world_size,
                              tensor_model_parallel_all_reduce)
from vllm.model_executor.layers.layernorm import RMSNorm
from vllm.model_executor.layers.linear import (ReplicatedLinear,
                                               QKVParallelLinear,
                                               RowParallelLinear)
from vllm.model_executor.layers.quantization import QuantizationConfig
from vllm.model_executor.layers.rotary_embedding import get_rope
from vllm.model_executor.models.utils import extract_layer_index


class GQAAttention(nn.Module):

    def __init__(
        self,
        config: PretrainedConfig,
        hidden_size: int,
        num_heads: int,
        num_kv_heads: int,
        rope_theta: float = 10000,
        rope_scaling: Optional[dict[str, Any]] = None,
        max_position_embeddings: int = 8192,
        qkv_bias: bool = False,
        o_bias: bool = False,
        cache_config: Optional[CacheConfig] = None,
        quant_config: Optional[QuantizationConfig] = None,
        prefix: str = "",
    ) -> None:
        super().__init__()
        self.layer_idx = extract_layer_index(prefix)
        self.hidden_size = hidden_size
        self.no_rope_layers = config.no_rope_layers
        self.nope = self.no_rope_layers[self.layer_idx] == 0
        self.use_qk_norm = config.use_qk_norm and not self.nope
        self.tp_size = get_tensor_model_parallel_world_size()
        self.total_num_heads = num_heads
        assert self.total_num_heads % self.tp_size == 0
        self.num_heads = self.total_num_heads // self.tp_size
        self.total_num_kv_heads = num_kv_heads
        if self.total_num_kv_heads >= self.tp_size:
            # Number of KV heads is greater than TP size, so we partition
            # the KV heads across multiple tensor parallel GPUs.
            assert self.total_num_kv_heads % self.tp_size == 0
        else:
            # Number of KV heads is less than TP size, so we replicate
            # the KV heads across multiple tensor parallel GPUs.
            assert self.tp_size % self.total_num_kv_heads == 0
        self.num_kv_heads = max(1, self.total_num_kv_heads // self.tp_size)
        self.head_dim = config.head_dim or (hidden_size // self.total_num_heads)
        self.q_size = self.num_heads * self.head_dim
        self.kv_size = self.num_kv_heads * self.head_dim
        self.scaling = self.head_dim**-0.5
        # TODO: attn_temperature_tuning should be a bool in huggingface
        self.attn_temperature_tuning = self.nope and \
            config.attn_temperature_tuning > 0
        self.floor_scale = getattr(config, "floor_scale", 8192.0)
        self.attn_scale = getattr(config, "attn_scale", 0.1)
        self.rope_theta = rope_theta
        self.max_position_embeddings = max_position_embeddings

        self.qkv_proj = QKVParallelLinear(hidden_size,
                                          self.head_dim,
                                          self.total_num_heads,
                                          self.total_num_kv_heads,
                                          bias=qkv_bias,
                                          quant_config=quant_config,
                                          prefix=f"{prefix}.qkv_proj")

        self.o_proj = RowParallelLinear(self.total_num_heads * self.head_dim,
                                        hidden_size,
                                        bias=o_bias,
                                        reduce_results=False,
                                        quant_config=quant_config,
                                        prefix=f"{prefix}.o_proj")
        self.rotary_emb = get_rope(
            self.head_dim,
            rotary_dim=self.head_dim,
            max_position=max_position_embeddings,
            base=rope_theta,
            rope_scaling=rope_scaling,
        )
        self.attn = Attention(self.num_heads,
                              self.head_dim,
                              self.scaling,
                              num_kv_heads=self.num_kv_heads,
                              cache_config=cache_config,
                              quant_config=quant_config,
                              prefix=f"{prefix}.attn")

        self.q_norm = RMSNorm(self.head_dim, eps=config.rms_norm_eps) if self.use_qk_norm else None
        self.k_norm = RMSNorm(self.head_dim, eps=config.rms_norm_eps) if self.use_qk_norm else None

    def forward(
        self,
        positions: torch.Tensor,
        hidden_states: torch.Tensor,
    ) -> torch.Tensor:
        q, k, v = self._forward_comp_qkv(hidden_states)
        q, k = self._forward_comp_rope(positions, q, k)
        attn_output = self._forward_comp_attn(positions, q, k, v)
        output, _ = self._forward_comp_output(attn_output)
        output = self._forward_comm_output(output)
        return output

    def forward_generator(
        self,
        positions: torch.Tensor,
        hidden_states: torch.Tensor,
    ) -> torch.Tensor:
        # TODO yield位置待调整补充
        q, k, v = self._forward_comp_qkv(hidden_states)
        q, k = self._forward_comp_rope(positions, q, k)
        attn_output = self._forward_comp_attn(positions, q, k, v)
        output, _ = self._forward_comp_output(attn_output)
        output = self._forward_comm_output(output)
        yield output

    def _forward_comp_qkv(
        self,
        hidden_states: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        qkv, _ = self.qkv_proj(hidden_states)
        q, k, v = qkv.split([self.q_size, self.kv_size, self.kv_size], dim=-1)
        if self.q_norm and self.k_norm:
            # Add qk-norm
            q_by_head = q.view(*q.shape[:-1], q.shape[-1] // self.head_dim,
                               self.head_dim)
            q_by_head = self.q_norm(q_by_head)
            q = q_by_head.view(q.shape)

            k_by_head = k.view(*k.shape[:-1], k.shape[-1] // self.head_dim,
                               self.head_dim)
            k_by_head = self.k_norm(k_by_head)
            k = k_by_head.view(k.shape)
        return q, k, v

    def _forward_comp_rope(
        self,
        positions: torch.Tensor,
        q: torch.Tensor,
        k: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        return self.rotary_emb(positions, q, k)

    def _get_attn_scale(self, positions: torch.Tensor) -> torch.Tensor:
        floor = torch.floor((positions + 1.0) / self.floor_scale)
        attn_scale = torch.log(floor + 1.0) * self.attn_scale + 1.0
        return attn_scale.unsqueeze(-1)

    def _forward_comp_attn(
        self,
        positions: torch.Tensor,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor
    ) -> torch.Tensor:
        # We are applying temperature tuning (https://arxiv.org/abs/2501.19399)
        # to NoPE layers, where the inference-time temperature tuning function
        # is customized to not affect short context
        # while working at very long context
        # https://arxiv.org/abs/2501.19399
        #
        # We should apply temperature tuning between (after) rotary / QK norm
        # and (before) attention.
        if self.attn_temperature_tuning and self.nope:
            attn_scale = self._get_attn_scale(positions)
            q = (q * attn_scale).to(q.dtype)
        return self.attn(q, k, v)

    def _forward_comp_output(
        self,
        attn_output: torch.Tensor
    ) -> torch.Tensor:
        output, _ = self.o_proj(attn_output)
        return output

    def _forward_comm_output(
        self,
        output_parallel: torch.Tensor
    ) -> torch.Tensor:
        """
        Attention applies all reduce if tp size is larger than 1.
        reduce_results is Ture within Attention
        """
        output = output_parallel
        if self.tp_size > 1:
            output = tensor_model_parallel_all_reduce(output_parallel)
        return output
