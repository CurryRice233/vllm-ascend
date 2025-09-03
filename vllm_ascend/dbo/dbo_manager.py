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

from typing import List, Optional, Tuple, Union

import torch
from vllm.forward_context import get_forward_context

from vllm_ascend.ascend_config import DualBatchOverlapConfig
from vllm_ascend.attention.attention_v1 import AscendMetadata
from vllm_ascend.attention.mla_v1 import AscendMLAMetadata
from vllm_ascend.dbo.dbo_utils import split_v1_mla_attn, split_v1_gqa_attn, \
    split_micro_batches_tensors
from vllm_ascend.dbo.dbo_context import DualBatchOverlapContext


class DualBatchOverlapManager:
    """
    DBO feature
    """
    # global DBO_STREAMS list
    DBO_STREAMS: List[torch.npu.Stream] = []

    def __init__(self, dbo_config, layers):
        self.dbo_config = dbo_config
        self.layers = layers
        self.num_batch = dbo_config.num_micro_batches
        # batch -> context
        self.dbo_contexts: List[DualBatchOverlapContext] = []
        self.use_mla = False
        self.init_dbo_streams()
        self.init_dbo_contexts()

    def init_dbo_streams(self):
        """
        init global dbo streams according to num_batch,
        including add stream nums if not enough.
        """
        for i in range(max(self.num_batch - len(self.DBO_STREAMS), 0)):
            stream = torch.npu.Stream()
            self.DBO_STREAMS.append(stream)

    def init_dbo_contexts(self):
        """
        each layer contains a batch of contexts, each contexts use its own stream.
        different layers can reuse contexts.
        """
        for i in range(self.num_batch):
            dbo_context = DualBatchOverlapContext(self.DBO_STREAMS[i])
            self.dbo_contexts.append(dbo_context)

    def run_forward(
        self,
        positions: torch.Tensor,
        hidden_states: torch.Tensor,
        residual: Optional[torch.Tensor] = None,
        **kwargs
    ):
        """
        entrance
        """
        # get attn_metadata
        attn_metadata = get_forward_context().attn_metadata
        self.use_mla = isinstance(attn_metadata, AscendMLAMetadata)
        # set config
        medata_cls = AscendMLAMetadata if self.use_mla else AscendMetadata
        # split attn_metadata、input_tensors
        attn_metadata, [positions, hidden_states, residual] = \
            self._split_layer_inputs(attn_metadata, medata_cls, self.dbo_config, [positions, hidden_states, residual])
        # iterate dbo_contexts, equal to iterate layers
        for layer in self.layers:
            hidden_states, residual = self.run_generator(
                layer, attn_metadata, positions, hidden_states, residual, **kwargs)
        # merge outputs
        [hidden_states, residual] = self._merge_layer_outputs([hidden_states, residual])
        return hidden_states, residual

    def run_generator(
        self,
        layer,
        attn_metadata: Union[List[AscendMLAMetadata], List[AscendMetadata]],
        positions: List[torch.Tensor],
        hidden_states: List[torch.Tensor],
        residual: Union[List[torch.Tensor], List[None]],
        **kwargs
    ) -> Tuple[List[torch.Tensor], List[torch.Tensor]]:
        """
        init and invoke generator based on single layer
        """
        context_generators = []
        for i in range(len(positions)):
            self.dbo_contexts[i].origin_generator = layer.forward_generator
            # set attn_metadata
            self.dbo_contexts[i].attn_metadata = attn_metadata[i]
            # init generator
            context_generator = self.dbo_contexts[i].prepare_generator(positions[i], hidden_states[i], residual[i], **kwargs)
            context_generators.append(context_generator)
        outputs = None
        hidden_states, residual = [], []
        # invoke generator
        for stage_output in zip(*context_generators):
            outputs = stage_output
        # analysis res for next layer
        for output in outputs:
            hidden_states.append(output[0])
            residual.append(output[1])
        return hidden_states, residual

    def _split_layer_inputs(
        self,
        attn_metadata: Union[AscendMetadata, AscendMLAMetadata],
        _metadata_cls,
        dbo_config: DualBatchOverlapConfig,
        intput_tensors: List[Optional[torch.Tensor]],
    ) -> Tuple[Union[List[AscendMLAMetadata], List[AscendMetadata]], List[List[torch.Tensor]]]:
        """
        split input info to num_batch
        return: Single or List of Metadata, such as AscendMLAMetadata、AscendMetadata.
                List of input tensors
        """
        # split attn_metadata, return a list, non-splittable if len of list is one,
        # otherwise return len is same as num_batch
        # TODO: need to support num_batch > 2
        attn_metadata = self._split_attn_metadata(attn_metadata, _metadata_cls, dbo_config)
        # split other input_tensors
        intput_tensors = self._split_input_tensors(attn_metadata, intput_tensors)
        return attn_metadata, intput_tensors

    @staticmethod
    def _merge_layer_outputs(
        input_tensors: List[List[torch.Tensor]]
    ):
        """
        merge outputs such as hidden_state, residual
        """
        merged_tensors: List[Optional[torch.Tensor]] = []
        for tensors in input_tensors:
            if tensors is None or tensors[0] is None:
                merged_tensors.append(None)
            else:
                merged_tensors.append(torch.cat(tensors, dim=0))
        return merged_tensors

    def _split_attn_metadata(
        self,
        attn_metadata: Union[AscendMetadata, AscendMLAMetadata],
        _metadata_cls,
        dbo_config: DualBatchOverlapConfig
    ) -> Union[List[AscendMLAMetadata], List[AscendMetadata]]:
        if self.use_mla:
            return split_v1_mla_attn(attn_metadata, _metadata_cls, dbo_config)
        return split_v1_gqa_attn(attn_metadata, _metadata_cls, dbo_config)

    @staticmethod
    def _split_input_tensors(
        attn_metadata,
        intput_tensors: List[Optional[torch.Tensor]],
    ) -> List[List[torch.Tensor]]:
        """
        split other input tensors, such as positions, hidden_states, residual
        return: List[List[torch.Tensor]]
        """
        # none split, trans output format
        if len(attn_metadata) == 1:
            out_tensors = []
            for tensor in intput_tensors:
                out_tensors.append([tensor])
            return out_tensors
        # TODO: need to support num_batch > 2
        split_index = attn_metadata[0].slot_mapping.shape[0]
        input_tensors = split_micro_batches_tensors(intput_tensors,
                                                    split_index)
        return input_tensors
