import math
from collections.abc import Iterable
from typing import List, Optional, Set, Tuple, Union

import torch
from torch import nn
from transformers import Llama4Config, Llama4VisionModel
from transformers.models.llama4.modeling_llama4 import Llama4MultiModalProjector

from sglang.srt.layers.logits_processor import LogitsProcessor
from sglang.srt.layers.moe.fused_moe_triton import FusedMoE
from sglang.srt.layers.quantization import QuantizationConfig
from sglang.srt.layers.rotary_embedding import RotaryEmbedding
from sglang.srt.managers.mm_utils import (
    MultiModalityDataPaddingPatternMultimodalTokens,
    general_mm_embed_routine,
)
from sglang.srt.managers.schedule_batch import MultimodalDataItem, MultimodalInputs
from sglang.srt.model_executor.forward_batch_info import ForwardBatch
from sglang.srt.model_loader.weight_utils import default_weight_loader
from sglang.srt.utils import add_prefix, is_hpu

_is_hpu = is_hpu()


class Llama4VisionRotaryEmbedding(RotaryEmbedding):
    def __init__(
        self,
        config,
    ):
        head_size = config.hidden_size // config.num_attention_heads
        rotary_dim = config.hidden_size // config.num_attention_heads // 2
        max_position_embeddings = (config.image_size // config.patch_size) ** 2
        base = config.rope_theta
        is_neox_style = False
        dtype = torch.bfloat16
        super().__init__(
            head_size, rotary_dim, max_position_embeddings, base, is_neox_style, dtype
        )

    def _compute_inv_freq(self, base: Union[int, float]) -> torch.Tensor:
        inv_freqs = super()._compute_inv_freq(base)
        inv_freqs = inv_freqs[: (self.rotary_dim // 2)]
        return inv_freqs

    def _compute_cos_sin_cache(self) -> torch.Tensor:
        inv_freq = self._compute_inv_freq(self.base)

        # Number of image patches (plus one extra row for CLS token, for example)
        num_patches = self.max_position_embeddings
        img_idx = torch.arange(num_patches, dtype=torch.int32).reshape(num_patches, 1)
        img_idx = torch.cat([img_idx, img_idx[:1]], dim=0)
        img_idx[-1, -1] = -2  # set to ID_CLS_TOKEN
        num_patches_single_dim = int(math.sqrt(num_patches))
        frequencies_x = img_idx % num_patches_single_dim
        frequencies_y = img_idx // num_patches_single_dim

        freqs_x = (
            (frequencies_x + 1)[..., None] * inv_freq[None, None, :]
        ).repeat_interleave(2, dim=-1)
        freqs_y = (
            (frequencies_y + 1)[..., None] * inv_freq[None, None, :]
        ).repeat_interleave(2, dim=-1)

        # The slicing reduces the last dimension so
        # that ultimately we have one angle per 2D pair.
        freqs = torch.cat([freqs_x, freqs_y], dim=-1).float().contiguous()[..., ::2]
        freqs = freqs.masked_fill(img_idx.reshape(-1, 1, 1) < 0, 0)

        # cache = torch.stack([torch.cos(freqs), torch.sin(freqs)], dim=-1)

        # Compute cosine and sine for each angle.
        cos_vals = torch.cos(freqs)
        sin_vals = torch.sin(freqs)

        cache = torch.concat([cos_vals, sin_vals], dim=-1)
        return cache

    def forward(
        self, query: torch.Tensor, key: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        # Ensure the cache is on the right device.
        self.cos_sin_cache = self.cos_sin_cache.to(query.device)
        cos_cache, sin_cache = self.cos_sin_cache.chunk(2, dim=-1)

        query_2d = query.float().reshape(*query.shape[:-1], -1, 2)
        key_2d = key.float().reshape(*key.shape[:-1], -1, 2)

        # Reshape cos_cache and sin_cache to broadcast properly.
        cos_cache = cos_cache.view(1, cos_cache.shape[0], 1, cos_cache.shape[-1])
        sin_cache = sin_cache.view(1, sin_cache.shape[0], 1, sin_cache.shape[-1])

        # Separate the real and imaginary parts.
        q_real, q_imag = query_2d.unbind(-1)  # each: [17, 577, 8, 44]
        k_real, k_imag = key_2d.unbind(-1)  # each: [17, 577, 8, 44]

        # Manually apply the complex multiplication (rotation) using the trigonometric identities.
        # For a complex multiplication: (a+ib)*(c+id) = (ac - bd) + i(ad + bc)
        q_rotated_real = q_real * cos_cache - q_imag * sin_cache
        q_rotated_imag = q_real * sin_cache + q_imag * cos_cache

        k_rotated_real = k_real * cos_cache - k_imag * sin_cache
        k_rotated_imag = k_real * sin_cache + k_imag * cos_cache

        # Re-stack the rotated components into a last dimension of size 2.
        q_rotated = torch.stack([q_rotated_real, q_rotated_imag], dim=-1)
        k_rotated = torch.stack([k_rotated_real, k_rotated_imag], dim=-1)

        # Flatten the last two dimensions to match the original output shape.
        # Flatten back to the desired shape (e.g., collapse the last two dimensions).
        query_out = q_rotated.flatten(3)
        key_out = k_rotated.flatten(3)

        return query_out.type_as(query), key_out.type_as(key)


if _is_hpu:
    import transformers.models.llama4.modeling_llama4

    # Monkey patch the Llama4VisionRotaryEmbedding class
    transformers.models.llama4.modeling_llama4.Llama4VisionRotaryEmbedding = (
        Llama4VisionRotaryEmbedding
    )


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

        self.vision_model = Llama4VisionModel(config.vision_config)
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

        image_outputs = self.vision_model(pixel_values, output_hidden_states=False)
        image_features = image_outputs.last_hidden_state
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
            if _is_cpu:
                dim = self.language_model.config.original_total_num_kv_heads
            else:
                dim = self.language_model.config.num_key_value_heads
            loaded_weight = permute(loaded_weight, dim)
        elif ("wq" in modules or "q_proj" in modules) and modules[-1] == "weight":
            if _is_cpu:
                dim = self.language_model.config.original_num_attention_heads
            else:
                dim = self.language_model.config.num_attention_heads
            loaded_weight = permute(loaded_weight, dim)

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
            if not "vision" in name:
                name, loaded_weight = self.permute_qk_weight_for_rotary(
                    name, loaded_weight
                )

            for param_name, weight_name, shard_id in stacked_params_mapping:
                if weight_name not in name:
                    continue

                if "vision" in name:
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

    def set_eagle3_layers_to_capture(self, layer_ids: Optional[List[int]] = None):
        if hasattr(self.language_model, "set_eagle3_layers_to_capture"):
            self.language_model.set_eagle3_layers_to_capture(layer_ids)

    def get_embed_and_head(self):
        # For EAGLE3, we delegate to the language model which should have this method
        # If the language model doesn't have lm_head (like EAGLE3), we return None for head
        embed = self.language_model.get_embed()
        if hasattr(self.language_model, "get_embed_and_head"):
            return self.language_model.get_embed_and_head()
        elif hasattr(self.language_model, "lm_head"):
            return embed, self.language_model.lm_head.weight
        else:
            # For EAGLE3, head might not be needed
            return embed, None

    def set_embed_and_head(self, embed, head):
        if hasattr(self.language_model, "set_embed_and_head"):
            return self.language_model.set_embed_and_head(embed, head)
        else:
            # For EAGLE3, only set embed
            return self.language_model.set_embed(embed)

    def get_embed(self):
        return self.language_model.get_embed()

    def set_embed(self, embed):
        return self.language_model.set_embed(embed)


EntryClass = Llama4ForConditionalGeneration
