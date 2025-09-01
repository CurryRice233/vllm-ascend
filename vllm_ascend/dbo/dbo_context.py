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

import torch
from vllm.forward_context import get_forward_context


class DualBatchOverlapContext:

    def __init__(self, forward_generator, stream, attn_metadata):
        self.origin_generator = forward_generator
        self.generator_with_context = None
        self.stream = stream
        self.attn_metadata = attn_metadata
        self.old_stream = None
        self.old_attn_metadata = None

    def __enter__(self):
        """
        invoked by with
        """
        self.old_stream = torch.npu.current_stream()
        self.old_attn_metadata = get_forward_context().attn_metadata
        torch.npu.set_stream(self.stream)
        get_forward_context().attn_metadata = self.attn_metadata

    def __exit__(self, exc_type, exc_val, exc_tb):
        """invoked while exit with module"""
        torch.npu.set_stream(self.old_stream)
        get_forward_context().attn_metadata = self.old_attn_metadata

    def prepare_generator(self, *args, **kwargs):
        return self.set_context(self.origin_generator(*args, **kwargs))

    def set_context(self, generator):
        while True:
            try:
                with self:
                    yield next(generator)
            except StopIteration:
                break
