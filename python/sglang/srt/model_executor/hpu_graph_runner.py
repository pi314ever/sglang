# Copyright 2023-2024 SGLang Team
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
# ==============================================================================
"""Run the model with hpu graph."""

from __future__ import annotations

import logging
import math
import os
import time
from collections import namedtuple
from contextlib import contextmanager
from typing import TYPE_CHECKING

import torch
import tqdm

from sglang.srt.layers.layernorm import RMSNorm
from sglang.srt.layers.logits_processor import LogitsProcessorOutput
from sglang.srt.layers.pooler import EmbeddingPoolerOutput
from sglang.srt.layers.vocab_parallel_embedding import VocabParallelEmbedding
from sglang.srt.managers.mm_utils import MultimodalInputs
from sglang.srt.model_executor.forward_batch_info import (
    CaptureHiddenMode,
    ForwardBatch,
    ForwardMode,
)
from sglang.srt.utils import is_hpu

_is_hpu = is_hpu()
if _is_hpu:

    os.environ["PT_HPU_ENABLE_LAZY_COLLECTIVES"] = "true"

    # Temporarily disabled due to accuracy issue in feature
    os.environ["VLLM_FUSED_BLOCK_SOFTMAX_ADJUSTMENT"] = "false"

    from sglang.srt.hpu_utils import (
        PREFILL_BUCKET_STEP,
        SKIP_WARMUP,
        USE_CONTIGUOUS_PA,
        compute_hpu_attn_bias_decode,
        compute_hpu_attn_bias_prefill,
        get_decode_all_buckets,
        get_decode_batch_bucket,
        get_prefill_all_prefix_seq_len_buckets,
        get_prefill_all_seq_len_buckets,
        get_prefill_prefix_seq_len_bucket,
        get_prefill_seq_len_bucket,
        prepare_hpu_attn_bias_prefill,
        to_hpu_and_pad_1d,
    )

if TYPE_CHECKING:
    from sglang.srt.model_executor.model_runner import ModelRunner

logger = logging.getLogger(__name__)

HPUForwardBatchBase = namedtuple(
    "HPUForwardBatch",
    [
        "forward_mode",
        "batch_size",
        "input_ids",
        "out_cache_loc",
        "positions",
        "attn_bias",
        "q_seq_pos",
        "q_seq_idx",
        "kv_seq_pos",
        "kv_seq_idx",
        "valid_seq_len",
        "extend_seq_lens",
        "page_size",
        "block_list",
        "block_mapping",
        "block_groups",
        "block_usage",
        "attn_backend",
        "token_to_kv_pool",
        "use_contiguous_pa",
        "mm_inputs",
        "top_logprobs_nums",
        "token_ids_logprobs",
        "extend_seq_lens_cpu",
        "extend_logprob_start_lens_cpu",
        "extend_input_logprob_token_ids_gpu",
        "global_num_tokens_gpu",
        "dp_local_start_pos",
        "dp_local_num_tokens",
        "gathered_buffer",
        "global_num_tokens_for_logprob_cpu",
        "global_num_tokens_for_logprob_gpu",
        "input_embeds",
        "return_logprob",
        "padded_static_len",
        "capture_hidden_mode",
    ],
    defaults=[
        None,  # top_logprobs_nums
        None,  # token_ids_logprobs
        None,  # extend_seq_lens_cpu
        None,  # extend_logprob_start_lens_cpu
        None,  # extend_input_logprob_token_ids_gpu
        None,  # global_num_tokens_gpu
        None,  # dp_local_start_pos
        None,  # dp_local_num_tokens
        None,  # gathered_buffer
        None,  # global_num_tokens_for_logprob_cpu
        None,  # global_num_tokens_for_logprob_gpu
        None,  # input_embeds
        False,  # return_logprob
        -1,  # padded_static_len
        CaptureHiddenMode.NULL,  # capture_hidden_mode
    ],
)

HPUMultimodalInputs = namedtuple(
    "HPUMultimodalInputs",
    [field.name for field in MultimodalInputs.__dataclass_fields__.values()],
)


class HPUForwardBatch(HPUForwardBatchBase):

    def contains_mm_inputs(self):
        return self.mm_inputs is not None

    def merge_mm_inputs(self):
        return self.mm_inputs


def set_hpu_torch_compile_config():
    import torch._dynamo.config

    torch._dynamo.config.accumulated_cache_size_limit = 8192
    if hasattr(torch._dynamo.config, "cache_size_limit"):
        torch._dynamo.config.cache_size_limit = 8192


def create_hpu_forward_batch(forward_batch: ForwardBatch, model_runner: ModelRunner):
    assert (
        forward_batch.hpu_metadata is not None
    ), "Expected HPU Metadata for HPU forward batch"
    batch_size = forward_batch.batch_size
    page_size = model_runner.token_to_kv_pool_allocator.page_size
    if forward_batch.forward_mode.is_extend():
        prompt_lens = forward_batch.extend_seq_lens
        prefix_lens = forward_batch.extend_prefix_lens
        if sum(prefix_lens) > 0:
            max_prefix_len = get_prefill_prefix_seq_len_bucket(
                sum(prefix_lens), page_size
            )
            max_prompt_len = get_prefill_seq_len_bucket(sum(prompt_lens))
        else:
            max_prefix_len = 0
            max_prompt_len = get_prefill_seq_len_bucket(sum(prompt_lens))
        attn_bias, q_seq_pos, q_seq_idx, kv_seq_pos, kv_seq_idx = (
            prepare_hpu_attn_bias_prefill(
                prompt_lens=prompt_lens,
                max_prompt_len=max_prompt_len,
                prefix_lens=prefix_lens,
                max_prefix_len=max_prefix_len,
                dtype=model_runner.dtype,
            )
        )
        attn_bias = attn_bias.to("hpu")
        q_seq_pos = q_seq_pos.to("hpu")
        q_seq_idx = q_seq_idx.to("hpu")
        kv_seq_pos = kv_seq_pos.to("hpu")
        kv_seq_idx = kv_seq_idx.to("hpu")
        padding_len = max_prompt_len - sum(prompt_lens)
        max_prefill_seqs = model_runner.server_args.max_running_requests
        input_ids = to_hpu_and_pad_1d(forward_batch.input_ids, padding_len)
        positions = to_hpu_and_pad_1d(forward_batch.positions, padding_len)
        valid_seq_len = prompt_lens.sum().to("hpu", dtype=torch.int64)
        extend_seq_lens_padded = to_hpu_and_pad_1d(
            forward_batch.extend_seq_lens, max_prefill_seqs - batch_size
        )
        extend_seq_lens_hpu = torch.tensor(
            forward_batch.extend_seq_lens_cpu, device="hpu", dtype=torch.int32
        )
        extend_seq_lens_hpu_padded = to_hpu_and_pad_1d(
            extend_seq_lens_hpu, max_prefill_seqs - batch_size
        )
        extend_logprob_start_lens_hpu = torch.tensor(
            forward_batch.extend_logprob_start_lens_cpu, device="hpu", dtype=torch.int32
        )
        extend_logprob_start_lens_hpu_padded = to_hpu_and_pad_1d(
            extend_logprob_start_lens_hpu, max_prefill_seqs - batch_size
        )
        out_cache_loc = to_hpu_and_pad_1d(forward_batch.out_cache_loc, padding_len)
        batch_size = 1
        block_list = (
            None
            if forward_batch.hpu_metadata is None
            else forward_batch.hpu_metadata.block_list.to("hpu")
        )
        block_mapping = None
        block_groups = None
        block_usage = None
        use_contiguous_pa = (
            False
            if forward_batch.hpu_metadata is None
            else forward_batch.hpu_metadata.use_contiguous_pa
        )
    else:
        padded_batch_size = get_decode_batch_bucket(batch_size)
        padding_len = padded_batch_size - batch_size
        input_ids = to_hpu_and_pad_1d(
            forward_batch.input_ids.to(torch.int64), padding_len
        )
        positions = to_hpu_and_pad_1d(
            forward_batch.positions.to(torch.int64), padding_len
        )
        valid_seq_len = torch.ones(padded_batch_size, dtype=torch.int64, device="hpu")
        out_cache_loc = to_hpu_and_pad_1d(forward_batch.out_cache_loc, padding_len)
        batch_size = padded_batch_size
        attn_bias = compute_hpu_attn_bias_decode(
            page_size, forward_batch.hpu_metadata.block_usage, model_runner.dtype
        )

        q_seq_idx = None
        q_seq_pos = None
        kv_seq_idx = None
        kv_seq_pos = None
        extend_seq_lens_padded = None
        block_list = forward_batch.hpu_metadata.block_list.to("hpu")
        block_mapping = forward_batch.hpu_metadata.block_mapping.to("hpu")
        block_groups = forward_batch.hpu_metadata.block_groups.to("hpu")
        block_usage = forward_batch.hpu_metadata.block_usage.to("hpu")
        use_contiguous_pa = forward_batch.hpu_metadata.use_contiguous_pa

        extend_seq_lens_hpu_padded = None
        extend_logprob_start_lens_hpu_padded = None

    if forward_batch.contains_mm_inputs():
        if forward_batch.contains_audio_inputs():
            raise NotImplementedError(f"Audio inputs are not supported yet")

        mm_inputs = forward_batch.merge_mm_inputs()
        mm_inputs = HPUMultimodalInputs(**mm_inputs.__dict__)
    else:
        mm_inputs = None

    return HPUForwardBatch(
        forward_mode=forward_batch.forward_mode,
        batch_size=batch_size,
        input_ids=input_ids,
        out_cache_loc=out_cache_loc,
        positions=positions,
        attn_bias=attn_bias,
        q_seq_pos=q_seq_pos,
        q_seq_idx=q_seq_idx,
        kv_seq_pos=kv_seq_pos,
        kv_seq_idx=kv_seq_idx,
        valid_seq_len=valid_seq_len,
        extend_seq_lens=extend_seq_lens_padded,
        page_size=page_size,
        block_list=block_list,
        block_mapping=block_mapping,
        block_groups=block_groups,
        block_usage=block_usage,
        attn_backend=forward_batch.attn_backend,
        token_to_kv_pool=forward_batch.token_to_kv_pool,
        use_contiguous_pa=use_contiguous_pa,
        mm_inputs=mm_inputs,
        return_logprob=forward_batch.return_logprob,
        top_logprobs_nums=forward_batch.top_logprobs_nums,
        token_ids_logprobs=forward_batch.token_ids_logprobs,
        extend_seq_lens_cpu=extend_seq_lens_hpu_padded,
        extend_logprob_start_lens_cpu=extend_logprob_start_lens_hpu_padded,
        extend_input_logprob_token_ids_gpu=forward_batch.extend_input_logprob_token_ids_gpu,
        global_num_tokens_gpu=forward_batch.global_num_tokens_gpu,
        dp_local_start_pos=forward_batch.dp_local_start_pos,
        dp_local_num_tokens=forward_batch.dp_local_num_tokens,
        gathered_buffer=forward_batch.gathered_buffer,
        global_num_tokens_for_logprob_cpu=forward_batch.global_num_tokens_for_logprob_cpu,
        global_num_tokens_for_logprob_gpu=forward_batch.global_num_tokens_for_logprob_gpu,
    )


def create_hpu_dummy_batch_prefill(
    prefix_len,
    prompt_len,
    dtype,
    page_size,
    max_running_requests,
    attn_backend,
    token_to_kv_pool,
    disable_prefix_cache=False,
):
    seq_len = prefix_len + prompt_len
    return HPUForwardBatch(
        forward_mode=ForwardMode.EXTEND,
        batch_size=1,
        input_ids=torch.zeros(prompt_len, dtype=torch.int64, device="hpu"),
        out_cache_loc=torch.arange(prompt_len, dtype=torch.int64, device="hpu"),
        positions=torch.zeros(prompt_len, dtype=torch.int64, device="hpu"),
        attn_bias=torch.zeros(1, 1, prompt_len, seq_len, dtype=dtype, device="hpu"),
        q_seq_pos=torch.zeros(1, prompt_len, dtype=torch.int64, device="hpu"),
        q_seq_idx=torch.zeros(1, prompt_len, dtype=torch.int64, device="hpu"),
        kv_seq_pos=torch.zeros(1, seq_len, dtype=torch.int64, device="hpu"),
        kv_seq_idx=torch.zeros(1, seq_len, dtype=torch.int64, device="hpu"),
        valid_seq_len=torch.ones((), dtype=torch.int64, device="hpu"),
        extend_seq_lens=torch.ones(
            max_running_requests,
            dtype=torch.int32,
            device="hpu",
        ),
        extend_seq_lens_cpu=torch.ones(
            max_running_requests,
            dtype=torch.int32,
            device="hpu",
        ),
        extend_logprob_start_lens_cpu=torch.ones(
            max_running_requests,
            dtype=torch.int32,
            device="hpu",
        ),
        page_size=page_size,
        block_list=torch.zeros(
            prefix_len // page_size, dtype=torch.int64, device="hpu"
        ),
        block_mapping=None,
        block_groups=None,
        block_usage=None,
        attn_backend=attn_backend,
        token_to_kv_pool=token_to_kv_pool,
        use_contiguous_pa=USE_CONTIGUOUS_PA and disable_prefix_cache,
        mm_inputs=None,
    )


def create_hpu_dummy_batch_decode(
    batch_size,
    block_num,
    dtype,
    page_size,
    attn_backend,
    token_to_kv_pool,
    disable_prefix_cache=False,
):
    return HPUForwardBatch(
        forward_mode=ForwardMode.DECODE,
        batch_size=batch_size,
        input_ids=torch.zeros(batch_size, dtype=torch.int64, device="hpu"),
        out_cache_loc=torch.zeros(batch_size, dtype=torch.int64, device="hpu"),
        positions=torch.zeros(batch_size, dtype=torch.int64, device="hpu"),
        attn_bias=torch.zeros(block_num, page_size, dtype=dtype, device="hpu"),
        q_seq_pos=None,
        q_seq_idx=None,
        kv_seq_pos=None,
        kv_seq_idx=None,
        valid_seq_len=torch.ones(batch_size, dtype=torch.int64, device="hpu"),
        extend_seq_lens=None,
        page_size=page_size,
        block_list=torch.zeros(block_num, dtype=torch.int64, device="hpu"),
        block_mapping=torch.zeros(block_num, batch_size, dtype=dtype, device="hpu"),
        block_groups=torch.zeros(block_num, dtype=torch.int64, device="hpu"),
        block_usage=torch.zeros(block_num, dtype=dtype, device="hpu"),
        attn_backend=attn_backend,
        token_to_kv_pool=token_to_kv_pool,
        use_contiguous_pa=USE_CONTIGUOUS_PA and disable_prefix_cache,
        mm_inputs=None,
    )


class HPUAdapter:

    def __init__(self, model, dtype) -> None:
        self.model = model
        self.dtype = dtype

    def __getattr__(self, name):
        return getattr(self.model, name)

    def forward(self, *args, **kwargs):
        assert len(args) == 3, "Only three arguments are supported"
        input_batch = args[2]
        if input_batch.forward_mode.is_extend():
            input_batch.attn_bias.copy_(
                compute_hpu_attn_bias_prefill(
                    input_batch.q_seq_pos,
                    input_batch.q_seq_idx,
                    input_batch.kv_seq_pos,
                    input_batch.kv_seq_idx,
                    self.dtype,
                )
            )
        return self.model(*args, **kwargs)

    def __call__(self, *args, **kwargs):
        return self.forward(*args, **kwargs)


class HPUGraphRunner:
    """A HPUGraphRunner runs the forward pass of a model with HPU graph and torch.compile."""

    def __init__(self, model_runner: ModelRunner):
        # Parse args
        self.model_runner = model_runner
        import habana_frameworks.torch as htorch
        import vllm_hpu_extension.environment as environment

        environment._VLLM_VALUES["model_type"] = (
            model_runner.model_config.hf_config.model_type
        )
        self.warmup_time = None

        self.is_lazy = 1 if htorch.utils.internal.is_lazy() else 0
        if self.is_lazy:
            hidden_layer_markstep_interval = int(
                os.getenv("SGLANG_CONFIG_HIDDEN_LAYERS", "1")
            )
            modify_model_layers(
                self.model_runner.model,
                get_target_layer_suffix_list(
                    model_runner.model_config.hf_config.model_type
                ),
                hidden_layer_markstep_interval,
            )

            self.model = htorch.hpu.wrap_in_hpu_graph(
                HPUAdapter(self.model_runner.model, self.model_runner.dtype),
                disable_tensor_cache=True,
            )
        elif self.model_runner.server_args.enable_torch_compile:
            set_hpu_torch_compile_config()
            self.regional_compilation_layers_list = [RMSNorm, VocabParallelEmbedding]
            self.model = HPUAdapter(self.model_runner.model, self.model_runner.dtype)
            compile_level = self.model_runner.server_args.regional_compile_level
            if compile_level == 0:  # root
                self.model = torch.compile(
                    self.model, backend="hpu_backend", dynamic=False
                )
                logger.info("Compiled the full model with torch.compile")
            elif compile_level == 1:  # 1 level below the root in tree
                self._regional_compilation(self.model)
                logger.info(
                    "Compiled all nodes 1 level below the root with torch.compile (regional mode)"
                )
            else:  # 2 levels below the root in tree
                self._compile_leaf_modules(self.model)
                logger.info(
                    "Compiled all leaf modules with torch.compile (regional mode)"
                )
        else:
            self.model = HPUAdapter(self.model_runner.model, self.model_runner.dtype)
            logger.info("Running on Eager mode.")

        # Capture
        if not SKIP_WARMUP:
            try:
                with self.model_capture_mode(), torch._dynamo.utils.disable_cache_limit():
                    logger.info(
                        "Begin to capture hpu graph, you can use `export SGLANG_HPU_SKIP_WARMUP=true` to skip this step."
                    )
                    time_start = time.perf_counter()
                    self.capture()
                    time_end = time.perf_counter()
                    self.warmup_time = time_end - time_start
                    logger.info(f"Capture hpu graph time: {self.warmup_time} seconds")
                    logger.info("Capture hpu graph success")
            except RuntimeError as e:
                raise Exception(f"Capture hpu graph failed: {e}\n")

    def _regional_compilation(self, module, parent_module=None, module_name=None):
        if any(
            isinstance(module, layer) for layer in self.regional_compilation_layers_list
        ):
            if parent_module is not None:
                self._compile_region(parent_module, module_name, module)

        elif isinstance(module, torch.nn.ModuleList):
            for child_name, child_module in module.named_children():
                for submodule_name, submodule in child_module.named_children():
                    self._compile_region(child_module, submodule_name, submodule)
        else:
            for children_name, children_module in module.named_children():
                self._regional_compilation(children_module, module, children_name)

    # TODO: Combine this function and `_regional_compilation` into one function using a simpler Level Order Traversal algorithm
    def _compile_leaf_modules(self, module, parent_module=None, module_name=None):
        """Recursively compile individual leaf modules while preserving module structure (op_level)."""
        if isinstance(module, (torch.nn.ModuleList, torch.nn.Sequential)):
            for name, child in module.named_children():
                self._regional_compilation_op_level(child, module, name)
            return

        if len(list(module.children())) == 0:
            logger.info(f"Compiling leaf module: {module_name}")
            compiled_module = torch.compile(
                module, backend="hpu_backend", dynamic=False
            )
            if parent_module is not None and module_name is not None:
                setattr(parent_module, module_name, compiled_module)
            return

        # For non-leaf modules, recursively compile children first
        for name, child in module.named_children():
            self._regional_compilation_op_level(child, module, name)

    def _compile_region(self, model, name, module):
        module = torch.compile(module, backend="hpu_backend", dynamic=False)
        setattr(model, name, module)

    @contextmanager
    def model_capture_mode(self):
        yield

    def can_run(self, forward_batch: ForwardBatch):
        return True

    def capture(self):
        # prefill
        time_start = time.perf_counter()
        max_prefill_tokens = self.model_runner.server_args.max_prefill_tokens
        prefill_seq_len_buckets = get_prefill_all_seq_len_buckets()
        prefill_step = PREFILL_BUCKET_STEP
        if self.model_runner.server_args.disable_radix_cache:
            prefill_prefix_len_buckets = [0]
        else:
            prefill_prefix_len_buckets = get_prefill_all_prefix_seq_len_buckets()
        for prompt_len in prefill_seq_len_buckets:
            for prefix_len in prefill_prefix_len_buckets:
                # add some head room
                if prefix_len + prompt_len > max_prefill_tokens + prefill_step:
                    continue
                self.capture_prefill(prefix_len, prompt_len)
        time_end = time.perf_counter()
        logger.info(f"Capture prefill time: {time_end - time_start} seconds")

        # decode
        if self.model_runner.is_generation:
            time_start = time.perf_counter()
            all_buckets = get_decode_all_buckets()
            for batch_size, seq_len in all_buckets:
                self.capture_decode(batch_size, seq_len)
            time_end = time.perf_counter()
            logger.info(f"Capture decode time: {time_end - time_start} seconds")

    def capture_prefill(self, prefix_len, prompt_len):
        logger.info(
            f"Capture prefill with prefix_len: {prefix_len} and prompt_len: {prompt_len}"
        )
        forward_batch = create_hpu_dummy_batch_prefill(
            prefix_len,
            prompt_len,
            self.model_runner.dtype,
            self.model_runner.token_to_kv_pool_allocator.page_size,
            self.model_runner.server_args.max_running_requests,
            self.model_runner.attn_backend,
            self.model_runner.token_to_kv_pool,
            disable_prefix_cache=self.model_runner.server_args.disable_radix_cache,
        )

        self.model_runner.attn_backend.init_forward_metadata(forward_batch)
        for i in range(3):
            self.model.forward(
                forward_batch.input_ids, forward_batch.positions, forward_batch
            )

    def capture_decode(self, batch_size, block_num):
        logger.info(
            f"Capture decode with batch_size: {batch_size} and block_num: {block_num}"
        )
        page_size = self.model_runner.token_to_kv_pool_allocator.page_size
        forward_batch = create_hpu_dummy_batch_decode(
            batch_size,
            block_num,
            self.model_runner.dtype,
            page_size,
            self.model_runner.attn_backend,
            self.model_runner.token_to_kv_pool,
            disable_prefix_cache=self.model_runner.server_args.disable_radix_cache,
        )
        self.model_runner.attn_backend.init_forward_metadata(forward_batch)
        for i in range(3):
            self.model.forward(
                forward_batch.input_ids, forward_batch.positions, forward_batch
            )

    def _forward(self, forward_batch: ForwardBatch):
        import habana_frameworks.torch as htorch

        forward_batch_hpu = create_hpu_forward_batch(forward_batch, self.model_runner)
        results = self.model.forward(
            forward_batch_hpu.input_ids, forward_batch_hpu.positions, forward_batch_hpu
        )
        htorch.core.mark_step()
        if isinstance(results, LogitsProcessorOutput):
            output = LogitsProcessorOutput(
                next_token_logits=results.next_token_logits.clone()[
                    : forward_batch.batch_size
                ],
                hidden_states=(
                    results.hidden_states.clone()[: forward_batch.batch_size]
                    if results.hidden_states is not None
                    else None
                ),
                input_token_logprobs=(
                    results.input_token_logprobs.clone()
                    if results.input_token_logprobs is not None
                    else None
                ),
                input_top_logprobs_val=(
                    results.input_top_logprobs_val
                    if results.input_top_logprobs_val is not None
                    else None
                ),
                input_top_logprobs_idx=(
                    results.input_top_logprobs_idx
                    if results.input_top_logprobs_idx is not None
                    else None
                ),
                input_token_ids_logprobs_val=(
                    results.input_token_ids_logprobs_val
                    if results.input_token_ids_logprobs_val is not None
                    else None
                ),
                input_token_ids_logprobs_idx=(
                    results.input_token_ids_logprobs_idx
                    if results.input_token_ids_logprobs_idx is not None
                    else None
                ),
            )
        elif isinstance(results, EmbeddingPoolerOutput):
            output = EmbeddingPoolerOutput(
                embeddings=results.embeddings.clone()[: forward_batch.batch_size]
            )
        return output

    def replay(
        self, forward_batch: ForwardBatch, skip_attn_backend_init: bool = False
    ) -> LogitsProcessorOutput:
        if not skip_attn_backend_init:
            self.model_runner.attn_backend.init_forward_metadata(forward_batch)
        return self._forward(forward_batch)


def get_target_layer_suffix_list(model_type) -> list[str]:
    # This sets the suffix for the hidden layer name, which is controlled by
    # SGLANG_CONFIG_HIDDEN_LAYERS. The default suffix is "DecoderLayer," which is
    # applicable for most language models such as LLaMA, Qwen, and BART. If the
    # model's decoder layer name differs from the default, it will need to
    # be specified here.
    decoder_layer_table = {
        "gpt_bigcode": "BigCodeBlock",
    }

    return [decoder_layer_table.get(model_type, "DecoderLayer"), "EncoderLayer"]


def modify_model_layers(
    module: torch.nn.Module, suffix_list: list[str], n=1, counter=None
):
    """Currently add mark_step at the end of specified layers."""
    import habana_frameworks.torch as htorch

    def forward_hook(module, args, output):
        htorch.core.mark_step()
        return output

    if counter is None:
        counter = [0]

    for child_name, child_module in module.named_children():
        if any(
            child_module.__class__.__name__.endswith(layer) for layer in suffix_list
        ):
            counter[0] += 1
            if counter[0] % n == 0:
                child_module.register_forward_hook(forward_hook)
        else:
            modify_model_layers(child_module, suffix_list, n, counter)
