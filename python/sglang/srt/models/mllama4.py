import math
from collections.abc import Iterable
from typing import List, Optional, Set, Tuple

import torch
from torch import nn
from transformers.models.llama4.configuration_llama4 import (
    Llama4Config,
    Llama4VisionConfig,
)
from transformers.models.llama4.modeling_llama4 import Llama4MultiModalProjector

from sglang.srt.distributed.parallel_state import get_tensor_model_parallel_world_size
from sglang.srt.layers.attention.vision import QKV_BACKEND_IMPL, info_once
from sglang.srt.layers.linear import (
    ColumnParallelLinear,
    QKVParallelLinear,
    RowParallelLinear,
)
from sglang.srt.layers.logits_processor import LogitsProcessor
from sglang.srt.layers.moe.fused_moe_triton import FusedMoE
from sglang.srt.layers.quantization import QuantizationConfig
from sglang.srt.layers.rotary_embedding import get_rope
from sglang.srt.managers.mm_utils import (
    MultiModalityDataPaddingPatternMultimodalTokens,
    general_mm_embed_routine,
)
from sglang.srt.managers.schedule_batch import (
    MultimodalDataItem,
    MultimodalInputs,
    global_server_args_dict,
)
from sglang.srt.model_executor.forward_batch_info import ForwardBatch
from sglang.srt.model_loader.weight_utils import default_weight_loader
from sglang.srt.utils import add_prefix


class Llama4UnfoldConvolution(nn.Module):
    def __init__(
        self,
        config: Llama4VisionConfig,
        quant_config: Optional[QuantizationConfig] = None,
        prefix: str = "",
    ):
        super().__init__()
        kernel_size = config.patch_size
        if isinstance(kernel_size, int):
            kernel_size = (kernel_size, kernel_size)
        self.unfold = torch.nn.Unfold(kernel_size=kernel_size, stride=config.patch_size)
        self.linear = ColumnParallelLinear(
            config.num_channels * kernel_size[0] * kernel_size[1],
            config.hidden_size,
            bias=False,
            quant_config=quant_config,
            gather_output=True,
            prefix=add_prefix("linear", prefix),
        )

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        hidden_states = self.unfold(hidden_states)
        hidden_states = hidden_states.permute(0, 2, 1)
        hidden_states, _ = self.linear(hidden_states)
        return hidden_states


class Llama4VisionMLP(nn.Module):
    def __init__(
        self,
        input_size: int,
        intermediate_size: int,
        output_size: int,
        bias: bool,
        output_activation: bool,
        quant_config: Optional[QuantizationConfig] = None,
        prefix: str = "",
    ):
        super().__init__()
        self.activation_fn = nn.GELU()  # ACT2FN[config.hidden_act]
        self.fc1 = ColumnParallelLinear(
            input_size,
            intermediate_size,
            bias=bias,
            quant_config=quant_config,
            prefix=add_prefix("fc1", prefix),
        )
        self.fc2 = RowParallelLinear(
            intermediate_size,
            output_size,
            bias=bias,
            quant_config=quant_config,
            prefix=add_prefix("fc2", prefix),
        )
        self.output_activation = output_activation

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        hidden_states, _ = self.fc1(hidden_states)
        hidden_states = self.activation_fn(hidden_states)
        hidden_states, _ = self.fc2(hidden_states)
        if self.output_activation:
            return self.activation_fn(hidden_states)
        return hidden_states


def pixel_shuffle(input_tensor, shuffle_ratio):
    # input_tensor: [batch_size, num_patches, channels]
    batch_size, num_patches, channels = input_tensor.shape
    patch_size = int(math.sqrt(num_patches))

    input_tensor = input_tensor.view(batch_size, patch_size, patch_size, -1)
    batch_size, height, width, channels = input_tensor.size()

    reshaped_tensor = input_tensor.view(
        batch_size, height, int(width * shuffle_ratio), int(channels / shuffle_ratio)
    )
    reshaped_tensor = reshaped_tensor.permute(0, 2, 1, 3).contiguous()

    reshaped_tensor = reshaped_tensor.view(
        batch_size,
        int(height * shuffle_ratio),
        int(width * shuffle_ratio),
        int(channels / (shuffle_ratio**2)),
    )
    reshaped_tensor = reshaped_tensor.permute(0, 2, 1, 3).contiguous()

    output_tensor = reshaped_tensor.view(batch_size, -1, reshaped_tensor.shape[-1])
    return output_tensor


class Llama4VisionPixelShuffleMLP(nn.Module):
    def __init__(
        self,
        config: Llama4VisionConfig,
        quant_config: Optional[QuantizationConfig] = None,
        prefix: str = "",
    ):
        super().__init__()
        self.pixel_shuffle_ratio = config.pixel_shuffle_ratio
        self.inner_dim = int(
            config.projector_input_dim // (self.pixel_shuffle_ratio**2)
        )
        self.output_dim = config.projector_output_dim
        self.mlp = Llama4VisionMLP(
            config.intermediate_size,
            config.projector_input_dim,
            config.projector_output_dim,
            config.multi_modal_projector_bias,
            output_activation=True,
            quant_config=quant_config,
            prefix=add_prefix("mlp", prefix),
        )

    def forward(self, encoded_patches: torch.Tensor) -> torch.Tensor:
        encoded_patches = pixel_shuffle(encoded_patches, self.pixel_shuffle_ratio)
        return self.mlp(encoded_patches)


class Llama4VisionAttention(nn.Module):
    def __init__(
        self,
        config: Llama4VisionConfig,
        quant_config: Optional[QuantizationConfig] = None,
        prefix: str = "",
    ):
        super().__init__()
        self.config = config
        self.tp_size = get_tensor_model_parallel_world_size()
        self.embed_dim = config.hidden_size
        self.num_heads = config.num_attention_heads
        self.head_dim = config.hidden_size // config.num_attention_heads
        self.num_key_value_groups = 1
        self.attention_dropout = config.attention_dropout
        self.num_local_heads = self.num_heads // self.tp_size
        self.q_size = self.num_local_heads * self.head_dim
        self.kv_size = self.num_local_heads * self.head_dim

        if global_server_args_dict["mm_attention_backend"] is None:
            qkv_backend = "sdpa"
            info_once(f"Multimodal attention backend not set. Use {qkv_backend}.")
        else:
            qkv_backend = global_server_args_dict["mm_attention_backend"]

        info_once(f"Using {qkv_backend} as multimodal attention backend.")

        self.qkv_backend = QKV_BACKEND_IMPL[qkv_backend](
            head_dim=self.head_dim,
            num_heads=self.num_local_heads,
            num_kv_heads=self.num_local_heads,
            dropout=0.0,
            flatten_batch=False,
            softmax_in_single_precision=False,
        )
        self.qkv_proj = QKVParallelLinear(
            self.embed_dim,
            self.head_dim,
            self.num_heads,
            bias=True,
            quant_config=quant_config,
            prefix=add_prefix("qkv_proj", prefix),
        )
        self.o_proj = RowParallelLinear(
            self.num_heads * self.head_dim,
            self.embed_dim,
            bias=False,
            quant_config=quant_config,
            prefix=add_prefix("o_proj", prefix),
        )
        self.rotary_emb = get_rope(
            self.head_dim,
            config.hidden_size // config.num_attention_heads // 2,
            (config.image_size // config.patch_size) ** 2,
            config.rope_theta,
            rope_scaling={"rope_type": "mllama4"},
            is_neox_style=False,
        )
        pass

    def forward(self, hidden_states: torch.Tensor):
        input_shape = hidden_states.shape[:-1]
        bsz = hidden_states.shape[0]

        qkv, _ = self.qkv_proj(hidden_states)
        q, k, v = qkv.split([self.q_size, self.kv_size, self.kv_size], dim=-1)

        q = q.view(q.shape[0], q.shape[1], self.num_local_heads, self.head_dim)
        k = k.view(k.shape[0], k.shape[1], self.num_local_heads, self.head_dim)
        q, k = self.rotary_emb(q, k)

        q = q.view(q.shape[0], q.shape[1], -1)
        k = k.view(k.shape[0], k.shape[1], -1)

        attn_output = self.qkv_backend(
            q,
            k,
            v,
            bsz,
            cu_seqlens=None,
            attention_mask=None,
        )

        attn_output = attn_output.reshape(*input_shape, -1).contiguous()
        attn_output, _ = self.o_proj(attn_output)
        return attn_output


class Llama4VisionEncoderLayer(nn.Module):
    def __init__(
        self,
        config: Llama4VisionConfig,
        quant_config: Optional[QuantizationConfig] = None,
        prefix="",
    ):
        super().__init__()
        self.hidden_size = config.hidden_size

        self.self_attn = Llama4VisionAttention(
            config, quant_config, prefix=add_prefix("self_attn", prefix)
        )
        self.mlp = Llama4VisionMLP(
            config.hidden_size,
            config.intermediate_size,
            config.hidden_size,
            True,
            False,
            quant_config,
            add_prefix("mlp", prefix),
        )

        self.input_layernorm = nn.LayerNorm(config.hidden_size)
        self.post_attention_layernorm = nn.LayerNorm(config.hidden_size)

    def forward(self, hidden_state: torch.Tensor):
        # Self Attention
        residual = hidden_state

        hidden_state = self.input_layernorm(hidden_state)

        hidden_state = self.self_attn(hidden_state)
        hidden_state = residual + hidden_state

        # Feed forward
        residual = hidden_state
        hidden_state = self.post_attention_layernorm(hidden_state)
        hidden_state = self.mlp(hidden_state)
        hidden_state = residual + hidden_state
        return hidden_state


class Llama4VisionEncoder(nn.Module):
    def __init__(
        self,
        config: Llama4VisionConfig,
        quant_config: Optional[QuantizationConfig] = None,
        prefix: str = "",
    ):
        super().__init__()
        self.config = config
        self.layers = nn.ModuleList(
            [
                Llama4VisionEncoderLayer(
                    config, quant_config, add_prefix(f"layers.{i}", prefix)
                )
                for i in range(config.num_hidden_layers)
            ]
        )
        self.gradient_checkpointing = False
        self.config = config

    def forward(
        self,
        hidden_states: torch.Tensor,
    ):
        for encoder_layer in self.layers:
            hidden_states = encoder_layer(hidden_state=hidden_states)

        return hidden_states


class Llama4VisionModel(nn.Module):
    def __init__(
        self,
        config: Llama4VisionConfig,
        quant_config: Optional[QuantizationConfig] = None,
        prefix: str = "",
    ):
        super().__init__()
        self.image_size = config.image_size
        self.patch_size = config.patch_size
        self.hidden_size = config.hidden_size
        self.num_channels = config.num_channels

        self.num_patches = (self.image_size // self.patch_size) ** 2 + 1
        self.scale = config.hidden_size**-0.5

        self.patch_embedding = Llama4UnfoldConvolution(config)

        self.class_embedding = nn.Parameter(self.scale * torch.randn(self.hidden_size))
        self.positional_embedding_vlm = nn.Parameter(
            self.scale * torch.randn(self.num_patches, self.hidden_size)
        )

        # layer norms
        self.layernorm_pre = nn.LayerNorm(self.hidden_size)
        self.layernorm_post = nn.LayerNorm(self.hidden_size)

        # encoders
        self.model = Llama4VisionEncoder(
            config, quant_config, add_prefix("model", prefix)
        )
        self.vision_adapter = Llama4VisionPixelShuffleMLP(config)

    def get_input_embeddings(self):
        """
        This function is used to fetch the first embedding layer to activate grads on inputs.
        """
        return self.patch_embedding

    def forward(
        self,
        pixel_values: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
    ):
        # num_concurrent_media and num_chunks are both currently 1
        batch_size_times_num_tiles, num_channels, height, width = pixel_values.shape
        num_concurrent_media = 1
        num_chunks = 1
        hidden_state = self.patch_embedding(pixel_values)
        _, num_patches, hidden_dim = hidden_state.shape

        # Add cls token
        hidden_state = hidden_state.reshape(
            batch_size_times_num_tiles * num_concurrent_media * num_chunks,
            num_patches,
            hidden_dim,
        )
        class_embedding = self.class_embedding.expand(
            hidden_state.shape[0], 1, hidden_state.shape[-1]
        )
        hidden_state = torch.cat([hidden_state, class_embedding], dim=1)
        num_patches += 1

        # Position embeddings
        hidden_state = hidden_state.reshape(
            batch_size_times_num_tiles * num_concurrent_media,
            num_chunks,
            num_patches,
            hidden_dim,
        )
        positional_embedding = self.positional_embedding_vlm.to(
            dtype=hidden_state.dtype, device=hidden_state.device
        )
        hidden_state = hidden_state + positional_embedding

        hidden_state = self.layernorm_pre(hidden_state)
        hidden_state = hidden_state.view(batch_size_times_num_tiles, -1, hidden_dim)
        hidden_state = self.model(hidden_state)
        hidden_state = self.layernorm_post(hidden_state)

        hidden_state = hidden_state[:, :-1, :]

        # now, we use Llama4VisionPixelShuffle + mlp to project embeddings
        hidden_state = self.vision_adapter(hidden_state)

        return hidden_state


class Llama4ForConditionalGeneration(nn.Module):
    packed_modules_mapping = {
        "qkv_proj": ["q_proj", "k_proj", "v_proj"],
        "gate_up_proj": ["gate_proj", "up_proj"],
    }

    def __init__(
        self,
        config: Llama4Config,
        quant_config: Optional[QuantizationConfig] = None,
        prefix: str = "",
    ):
        super().__init__()
        self.config = config
        self.quant_config = quant_config

        self.vision_model = Llama4VisionModel(
            config.vision_config, quant_config, add_prefix("vision_model", prefix)
        )
        self.multi_modal_projector = Llama4MultiModalProjector(config)

        # Initialize the language model
        from sglang.srt.models.llama4 import Llama4ForCausalLM

        self.language_model = Llama4ForCausalLM(
            config.text_config,
            quant_config=quant_config,
            prefix=add_prefix("language_model", prefix),
        )

        self.logits_processor = LogitsProcessor(config.text_config)

    def pad_input_ids(self, input_ids: List[int], mm_inputs: MultimodalInputs):
        # Get all special token IDs
        im_token_id: int = mm_inputs.im_token_id

        pattern = MultiModalityDataPaddingPatternMultimodalTokens([im_token_id])
        return pattern.pad_input_tokens(input_ids, mm_inputs)

    def get_image_feature(
        self,
        items: List[MultimodalDataItem],
    ) -> torch.Tensor:
        pixel_values = (
            torch.concat([item.pixel_values for item in items])
            .to(next(self.vision_model.parameters()).device)
            .type(next(self.vision_model.parameters()).dtype)
        )

        image_features = self.vision_model(pixel_values)
        vision_flat = image_features.view(-1, image_features.size(-1))
        projected_vision_flat = self.multi_modal_projector(vision_flat)
        return projected_vision_flat

    def forward(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        forward_batch: ForwardBatch,
        **kwargs: object,
    ) -> torch.Tensor:
        hs = general_mm_embed_routine(
            input_ids=input_ids,
            forward_batch=forward_batch,
            language_model=self.language_model,
            image_data_embedding_func=self.get_image_feature,
            positions=positions,
        )

        return hs

    def permute_qk_weight_for_rotary(
        self,
        name: str,
        loaded_weight: torch.Tensor,
    ) -> Tuple[str, torch.Tensor]:
        def permute(w: torch.Tensor, n_heads: int):
            attn_in = self.language_model.config.head_dim * n_heads
            attn_out = self.language_model.config.hidden_size

            return (
                w.view(n_heads, attn_in // n_heads // 2, 2, attn_out)
                .transpose(1, 2)
                .reshape(attn_in, attn_out)
            )

        modules = name.split(".")

        # rotary embeds should be sliced
        if ("wk" in modules or "k_proj" in modules) and modules[-1] == "weight":
            loaded_weight = permute(
                loaded_weight, self.language_model.config.num_key_value_heads
            )
        elif ("wq" in modules or "q_proj" in modules) and modules[-1] == "weight":
            loaded_weight = permute(
                loaded_weight, self.language_model.config.num_attention_heads
            )

        return name, loaded_weight

    def load_weights(self, weights: Iterable[Tuple[str, torch.Tensor]]) -> Set[str]:
        stacked_params_mapping = [
            # (param_name, shard_name, shard_id)
            (".self_attn.qkv_proj", ".self_attn.q_proj", "q"),
            (".self_attn.qkv_proj", ".self_attn.k_proj", "k"),
            (".self_attn.qkv_proj", ".self_attn.v_proj", "v"),
            (".shared_expert.gate_up_proj", ".shared_expert.gate_proj", 0),
            (".shared_expert.gate_up_proj", ".shared_expert.up_proj", 1),
            (".feed_forward.gate_up_proj", ".feed_forward.gate_proj", 0),
            (".feed_forward.gate_up_proj", ".feed_forward.up_proj", 1),
        ]

        params_dict = dict(self.named_parameters())

        num_experts = self.config.text_config.num_local_experts

        # Params for weights, fp8 weight scales, fp8 activation scales
        # (param_name, weight_name, expert_id, shard_id)
        expert_params_mapping = FusedMoE.make_expert_params_mapping(
            ckpt_gate_proj_name="gate_proj",
            ckpt_down_proj_name="down_proj",
            ckpt_up_proj_name="up_proj",
            num_experts=num_experts,
        )

        for name, loaded_weight in weights:
            if "vision" not in name:
                name, loaded_weight = self.permute_qk_weight_for_rotary(
                    name, loaded_weight
                )

            for param_name, weight_name, shard_id in stacked_params_mapping:
                if weight_name not in name:
                    continue

                if "vision" in name and "self_attn" not in name:
                    continue
                name = name.replace(weight_name, param_name)
                param = params_dict[name]
                weight_loader = param.weight_loader
                weight_loader(param, loaded_weight, shard_id)
                break
            else:
                if ".experts" in name:
                    # NOTE: llama4 fp8 has different weight format for experts
                    if (
                        "experts.gate_up_proj" not in name
                        and "experts.down_proj" not in name
                    ):
                        for mapping in expert_params_mapping:
                            param_name, weight_name, expert_id, shard_id = mapping
                            if weight_name not in name:
                                continue
                            name = name.replace(weight_name, param_name)
                            param = params_dict[name]
                            weight_loader = param.weight_loader
                            weight_loader(
                                param,
                                loaded_weight,
                                name,
                                shard_id=shard_id,
                                expert_id=expert_id,
                            )
                            break
                    else:
                        if ".gate_up_proj" in name:
                            name_list = [
                                name.replace(
                                    ".experts.gate_up_proj", ".experts.w13_weight"
                                )
                            ] * 2
                            loaded_weight_list = loaded_weight.chunk(2, dim=-1)
                            shard_id_list = ["w1", "w3"]
                        else:
                            name_list = [
                                name.replace(".experts.down_proj", ".experts.w2_weight")
                            ]
                            shard_id_list = ["w2"]
                            loaded_weight_list = [loaded_weight]
                        for name, loaded_weight, shard_id in zip(
                            name_list, loaded_weight_list, shard_id_list
                        ):
                            param = params_dict[name]
                            weight_loader = param.weight_loader
                            for expert_id in range(num_experts):
                                weight_loader(
                                    param,
                                    loaded_weight[expert_id].T,
                                    name,
                                    shard_id=shard_id,
                                    expert_id=expert_id,
                                )
                else:
                    # Skip loading extra bias for GPTQ models.
                    if name.endswith(".bias") and name not in params_dict:
                        continue
                    param = params_dict[name]
                    weight_loader = getattr(
                        param, "weight_loader", default_weight_loader
                    )
                    weight_loader(param, loaded_weight)


EntryClass = Llama4ForConditionalGeneration
