#
# Copyright (c) 2025 Huawei Technologies Co., Ltd. All Rights Reserved.
# This file is a part of the vllm-ascend project.
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
#

from copy import deepcopy
from typing import Optional, Union, List, Tuple, Any, Dict

import numpy as np
import torch

from vllm_ascend.ascend_config import DualBatchOverlapConfig
from vllm_ascend.attention.attention_v1 import AscendAttentionState, AscendMetadata
from vllm_ascend.attention.mla_v1 import AscendMLAMetadata


def compute_split_seq_index(
    query_lens: Optional[List[int]],
    attn_state: AscendAttentionState,
    num_tokens: int,
    imbalance_ratio: float = 0.1,
) -> List[int]:
    if attn_state != AscendAttentionState.DecodeOnly:
        assert query_lens is not None
        total_tokens = sum(query_lens)
        # the first index in last split
        tokens, split_index = 0, 0
        for value in query_lens:
            tokens += value
            split_index += 1
            if tokens >= total_tokens // 2:
                # check the current split index
                if abs(tokens -
                       total_tokens // 2) < total_tokens * imbalance_ratio:
                    return [tokens, split_index]
                # check the previous split index
                elif abs(tokens - total_tokens // 2 -
                         value) < total_tokens * imbalance_ratio:
                    return [tokens - value, split_index - 1]
                # fail to split if it is imbalanced
                # TODO: split tokens in seq
                else:
                    return [0, 0]
    else:
        tokens = num_tokens // 2
        return [tokens, tokens]
    return [0, 0]


def split_attn_tensor_type(
    input_tensor: Union[torch.Tensor, List[int]],
    index: int,
) -> Union[List[torch.Tensor], List[int]]:
    return [input_tensor[:index], input_tensor[index:]]


def split_attn_int_type(
    var: int,
    index: int,
) -> List[int]:
    return [min(var, index), max(var - index, 0)]


def check_if_split_valid(
    attn_metadata: Union[AscendMetadata, AscendMLAMetadata],
    dbo_config: DualBatchOverlapConfig,
) -> Tuple[bool, List[Any]]:
    if attn_metadata is None or dbo_config.num_micro_batches < 2:
        return False, [0, 0]
    if isinstance(attn_metadata, AscendMLAMetadata):
        num_tokens = attn_metadata.num_decode_tokens
    else:
        num_tokens = attn_metadata.num_actual_tokens
    [token_index, seq_index] = compute_split_seq_index(attn_metadata.query_lens, attn_metadata.attn_state,
                                                       num_tokens, dbo_config.imbalance_ratio)
    if token_index == 0 or seq_index == 0 or seq_index == len(attn_metadata.query_lens):
        return False, [0, 0]
    return True, [token_index, seq_index]


def split_common_attn_metadata(attn_metadata, token_index, seq_index):
    """common func for mla and gqa"""
    [slot_mapping_pre,
     slot_mapping_post] = split_attn_tensor_type(attn_metadata.slot_mapping,
                                                 token_index)
    query_start_loc_pre = query_start_loc_post = None
    if attn_metadata.query_start_loc is not None:
        query_start_loc_pre = attn_metadata.query_start_loc[:seq_index + 1]
        query_start_loc_post = deepcopy(
            attn_metadata.query_start_loc[seq_index:]
        ) - attn_metadata.query_start_loc[seq_index]
    [block_table_pre,
     block_table_post] = split_attn_tensor_type(attn_metadata.block_tables,
                                                seq_index)
    return [slot_mapping_pre, slot_mapping_post, query_start_loc_pre,
            query_start_loc_post, block_table_pre, block_table_post]


def split_common_attn_state_mask(attn_metadata, token_index, has_prefill, seq_lens_pre, seq_lens_post):
    """common func for mla and gqa"""
    if (attn_metadata.attn_state == AscendAttentionState.PrefillNoCache or
            attn_metadata.attn_state == AscendAttentionState.PrefillCacheHit):
        # the attn_mla kernel in torch npu only accept 128*128 attn mask
        attn_mask_pre = attn_mask_post = attn_metadata.attn_mask
        attn_state_pre = attn_state_post = attn_metadata.attn_state
    elif attn_metadata.attn_state == AscendAttentionState.DecodeOnly:
        # should be none in decode only state
        attn_mask_pre = attn_mask_post = attn_metadata.attn_mask
        attn_state_pre = attn_state_post = AscendAttentionState.DecodeOnly
    else:
        # chunked prefill
        assert attn_metadata.attn_mask is not None
        if has_prefill > 0:
            attn_state_pre = AscendAttentionState.ChunkedPrefill
            attn_mask_pre = attn_metadata.attn_mask[:token_index, :max(
                seq_lens_pre)].contiguous()
            attn_state_post = AscendAttentionState.ChunkedPrefill
            attn_mask_post = attn_metadata.attn_mask[
                             token_index:, :max(seq_lens_post)].contiguous()
        else:
            attn_state_pre = AscendAttentionState.DecodeOnly
            attn_mask_pre = None
            attn_state_post = AscendAttentionState.ChunkedPrefill
            attn_mask_post = attn_metadata.attn_mask[
                             token_index:, :max(seq_lens_post)].contiguous()
    return [attn_mask_pre, attn_state_pre, attn_mask_post, attn_state_post]


def split_mla_partial_attn(
    attn_metadata: AscendMLAMetadata,
    seq_index: int,
    token_index: int,
):
    query_start_loc_cpu = np.zeros(shape=(len(attn_metadata.query_lens) + 1,), dtype=int)
    np.cumsum(attn_metadata.query_lens, out=query_start_loc_cpu[1:])
    if attn_metadata.num_prefills > 0:
        prefill_query_start_loc = np.zeros(
            shape=(len(attn_metadata.prefill.query_lens) + 1,), dtype=int)
        np.cumsum(attn_metadata.prefill.query_lens,
                  out=prefill_query_start_loc[1:])

    [num_prefills_pre, num_prefills_post] = split_attn_int_type(
        attn_metadata.num_prefills, max(0, seq_index - attn_metadata.num_decodes))

    [num_decodes_pre, num_decodes_post] = split_attn_int_type(attn_metadata.num_decodes, seq_index)
    [num_decode_tokens_pre, num_decode_tokens_post] = split_attn_int_type(attn_metadata.num_decode_tokens, token_index)

    seq_lens = attn_metadata.prefill.seq_lens if attn_metadata.num_prefills > 0 else attn_metadata.decode.seq_lens
    [seq_lens_pre, seq_lens_post] = split_attn_tensor_type(seq_lens, seq_index)

    return [num_prefills_pre, num_prefills_post, num_decodes_pre, num_decodes_post,
            num_decode_tokens_pre, num_decode_tokens_post, seq_lens_pre, seq_lens_post, seq_lens]


def split_v1_mla_attn(
    attn_metadata: AscendMLAMetadata,
    _metadata_cls,
    dbo_config: DualBatchOverlapConfig,
) -> List[Any]:
    """entrance for mla"""
    flag, index_list = check_if_split_valid(attn_metadata, dbo_config)
    if not flag:
        return [attn_metadata]
    [token_index, seq_index] = index_list

    [num_prefills_pre, num_prefills_post, num_decodes_pre, num_decodes_post,
     num_decode_tokens_pre, num_decode_tokens_post, seq_lens_pre, seq_lens_post, seq_lens] = split_mla_partial_attn(
        attn_metadata, seq_index, token_index)

    [slot_mapping_pre, slot_mapping_post, query_start_loc_pre,
     query_start_loc_post, block_table_pre, block_table_post] = split_common_attn_metadata(
        attn_metadata, token_index, seq_index)

    [attn_mask_pre, attn_state_pre, attn_mask_post, attn_state_post] = split_common_attn_state_mask(
        attn_metadata, token_index, num_prefills_pre, seq_lens_pre, seq_lens_post)

    from vllm_ascend.attention.mla_v1 import (AscendMLADecodeMetadata,
                                              AscendMLAPrefillMetadata)
    if num_prefills_pre > 0:
        # split metadata.prefill
        [input_positions_pre, input_positions_post] = split_attn_tensor_type(
            attn_metadata.prefill.input_positions,
            token_index - attn_metadata.num_decode_tokens)
        [block_tables_pre, block_tables_post
         ] = split_attn_tensor_type(attn_metadata.prefill.block_table,
                                    seq_index - attn_metadata.num_decodes)
        [prefill_query_lens_pre, prefill_query_lens_post
         ] = split_attn_tensor_type(attn_metadata.prefill.query_lens,
                                    seq_index - attn_metadata.num_decodes)
        prefill_query_start_loc_pre = attn_metadata.prefill.query_start_loc[:seq_index + 1 - attn_metadata.num_decodes]
        prefill_query_start_loc_post = deepcopy(
            attn_metadata.prefill.query_start_loc[seq_index - attn_metadata.num_decodes:]
        ) - attn_metadata.prefill.query_start_loc[seq_index - attn_metadata.num_decodes]
        context_len_pre = seq_lens_pre[attn_metadata.num_decodes:]
        context_len_post = seq_lens_post
        prefill_max_query_len_pre = max(prefill_query_lens_pre)
        prefill_max_query_len_post = max(prefill_query_lens_post)
        prefill_pre = AscendMLAPrefillMetadata(
            attn_mask=attn_mask_pre,
            query_lens=prefill_query_lens_pre,
            seq_lens=seq_lens_pre,
            query_start_loc=prefill_query_start_loc_pre,
            input_positions=input_positions_pre,
            context_lens=context_len_pre,
            block_table=block_tables_pre,
            max_query_len=prefill_max_query_len_pre,
            max_seq_lens=context_len_pre.max().item(),
        )
        prefill_post = AscendMLAPrefillMetadata(
            attn_mask=attn_mask_post,
            query_lens=prefill_query_lens_post,
            seq_lens=seq_lens_post,
            query_start_loc=prefill_query_start_loc_post,
            input_positions=input_positions_post,
            context_lens=context_len_post,
            block_table=block_tables_post,
            max_query_len=prefill_max_query_len_post,
            max_seq_lens=context_len_post.max().item(),
        )
        decode_pre = attn_metadata.decode
        decode_post = None
    else:
        # prefill is None, split metadata.decode
        [input_positions_pre, input_positions_post] = split_attn_tensor_type(attn_metadata.decode.input_positions,
                                                                             token_index)
        [block_tables_pre, block_tables_post] = split_attn_tensor_type(attn_metadata.decode.block_table,
                                                                       seq_index)
        [decode_seq_lens_pre, decode_seq_lens_post] = split_attn_tensor_type(seq_lens, seq_index)
        decode_pre = AscendMLADecodeMetadata(
            input_positions=input_positions_pre,
            block_table=block_tables_pre,
            seq_lens=decode_seq_lens_pre,
            max_seq_lens=max(decode_seq_lens_pre),
            seq_lens_list=decode_seq_lens_pre.tolist(),
        )
        decode_post = AscendMLADecodeMetadata(
            input_positions=input_positions_post,
            block_table=block_tables_post,
            seq_lens=decode_seq_lens_post,
            max_seq_lens=max(decode_seq_lens_post),
            seq_lens_list=decode_seq_lens_post.tolist(),
        )
        prefill_pre = None
        prefill_post = attn_metadata.prefill
    # construct metadata
    attention_metadata_pre = _metadata_cls(
        num_actual_tokens=token_index,
        num_input_tokens=token_index,
        head_dim=attn_metadata.head_dim,
        slot_mapping=slot_mapping_pre,
        seq_lens=seq_lens_pre,
        query_start_loc=query_start_loc_pre,
        block_tables=block_table_pre,
        num_decodes=num_decodes_pre,
        num_prefills=num_prefills_pre,
        num_decode_tokens=num_decode_tokens_pre,
        attn_state=attn_state_pre,
        attn_mask=attn_mask_pre,
        prefill=prefill_pre,
        decode=decode_pre,
        enable_dbo_across_dp=attn_metadata.enable_dbo_across_dp,
    )
    attention_metadata_post = _metadata_cls(
        num_actual_tokens=attn_metadata.num_actual_tokens - token_index,
        num_input_tokens=attn_metadata.num_input_tokens - token_index,
        head_dim=attn_metadata.head_dim,
        slot_mapping=slot_mapping_post,
        seq_lens=seq_lens_post,
        query_start_loc=query_start_loc_post,
        block_tables=block_table_post,
        num_decodes=num_decodes_post,
        num_prefills=num_prefills_post,
        num_decode_tokens=num_decode_tokens_post,
        attn_mask=attn_mask_post,
        attn_state=attn_state_post,
        prefill=prefill_post,
        decode=decode_post,
        enable_dbo_across_dp=attn_metadata.enable_dbo_across_dp,
    )
    return [attention_metadata_pre, attention_metadata_post]


def split_gqa_partial_attn(
    attn_metadata: AscendMetadata,
    seq_index: int,
):
    [query_lens_pre, query_lens_post] = split_attn_tensor_type(attn_metadata.query_lens,
                                                               seq_index)
    [seq_lens_pre, seq_lens_post] = split_attn_tensor_type(attn_metadata.seq_lens, seq_index)

    max_query_len_pre = max_query_len_post = None
    if attn_metadata.max_query_len is not None:
        max_query_len_pre, max_query_len_post = max(query_lens_pre), max(query_lens_post)
    return [query_lens_pre, query_lens_post, seq_lens_pre, seq_lens_post, max_query_len_pre, max_query_len_post]


def split_v1_gqa_attn(
    attn_metadata: AscendMetadata,
    _metadata_cls,
    dbo_config: DualBatchOverlapConfig,
) -> List[Any]:
    """entrance for gqa"""
    flag, index_list = check_if_split_valid(attn_metadata, dbo_config)
    if not flag:
        return [attn_metadata]
    [token_index, seq_index] = index_list

    [query_lens_pre, query_lens_post, seq_lens_pre, seq_lens_post,
     max_query_len_pre, max_query_len_post] = split_gqa_partial_attn(attn_metadata, seq_index)

    [slot_mapping_pre, slot_mapping_post, query_start_loc_pre,
     query_start_loc_post, block_table_pre, block_table_post] = split_common_attn_metadata(
        attn_metadata, token_index, seq_index)

    has_prefill_pre, _ = torch.any(query_lens_pre > 1).item(), torch.any(
        query_lens_post > 1).item()

    [attn_mask_pre, attn_state_pre, attn_mask_post, attn_state_post] = split_common_attn_state_mask(
        attn_metadata, token_index, has_prefill_pre, seq_lens_pre, seq_lens_post)

    # construct metadata
    attention_metadata_pre = _metadata_cls(
        num_actual_tokens=token_index,
        block_tables=block_table_pre,
        query_start_loc=query_start_loc_pre,
        query_lens=query_lens_pre,
        seq_lens=seq_lens_pre,
        max_query_len=max_query_len_pre,
        slot_mapping=slot_mapping_pre,
        attn_state=attn_state_pre,
        attn_mask=attn_mask_pre,
        enable_dbo_across_dp=attn_metadata.enable_dbo_across_dp,
        is_only_prefill=attn_metadata.is_only_prefill,
    )

    attention_metadata_post = _metadata_cls(
        num_actual_tokens=attn_metadata.num_actual_tokens - token_index,
        block_tables=block_table_post,
        query_start_loc=query_start_loc_post,
        query_lens=query_lens_post,
        seq_lens=seq_lens_post,
        max_query_len=max_query_len_post,
        slot_mapping=slot_mapping_post,
        attn_state=attn_state_post,
        attn_mask=attn_mask_post,
        enable_dbo_across_dp=attn_metadata.enable_dbo_across_dp,
        is_only_prefill=attn_metadata.is_only_prefill,
    )

    return [attention_metadata_pre, attention_metadata_post]


def split_micro_batches_tensors(
    input_tensors: List[Optional[torch.Tensor]],
    split_index: int,
    keys: Optional[List[str]] = None
):
    if isinstance(input_tensors, list):
        micro_batches = []
        for tensor in input_tensors:
            if tensor is None:
                micro_batches.append([None, None])
            else:
                micro_batches.append(
                    [tensor[:split_index], tensor[split_index:]])
        return micro_batches
    elif isinstance(input_tensors, torch.Tensor):
        return [input_tensors[:split_index], input_tensors[split_index:]]
    elif input_tensors is None:
        return [None, None]
    elif isinstance(input_tensors, Dict):
        assert keys is not None
        micro_batches_pre = {}
        for key in keys:
            micro_batches_pre[key] = input_tensors[key][:split_index]
        micro_batches_post = {}
        for key in keys:
            micro_batches_post[key] = input_tensors[key][split_index:]
        return [micro_batches_pre, micro_batches_post]
    else:
        raise NotImplementedError
