from typing import Tuple, Union

import torch

from sglang.srt.lora.backend.base_backend import BaseLoRABackend
from sglang.srt.lora.utils import LoRABatchInfo


class TorchLoRABackend(BaseLoRABackend):
    def __init__(self, name: str, batch_info: LoRABatchInfo = None):
        super().__init__(name, batch_info)
        self.exploded_indices = None
        self.scalings = None

    def run_lora_a_sgemm(
        self, x: torch.Tensor, weights: torch.Tensor, *args, **kwargs
    ) -> torch.Tensor:
        """Run segment Gemm of lora a modules with current backend.

        Args:
             x: input matrix with shape (s, input_dim), here s is the sum of all sequence lengths
             weights: a set of lora weights with shape (num_lora, c * r, input_dim),
                      here r is lora rank, c is a multiplier for stacked modules (e.g., c=3 for qkv_proj, c=2 for gate_up_proj)
                      usually input_dim is much larger than r
        Returns:
             result with shape (s, c * r)
        """
        selected_loras = torch.index_select(weights, 0, self.exploded_indices)
        outputs = x.unsqueeze(-2) @ selected_loras.transpose(-1, -2)
        return outputs.reshape(-1, outputs.shape[-1])

    def run_lora_b_sgemm(
        self, x: torch.Tensor, weights: torch.Tensor, *args, **kwargs
    ) -> torch.Tensor:
        """Run segment Gemm of lora b modules with current backend.

        Args:
             x: input matrix with shape (s, r), here s is the sum of all sequence lengths, r is lora rank
             weights: a set of lora weights with shape (num_lora, output_dim, r)
                      usually output_dim is much larger than r
        Returns:
             result with shape (s, output_dim)
        """
        selected_loras = torch.index_select(weights, 0, self.exploded_indices)
        outputs = x.unsqueeze(-2) @ selected_loras.transpose(-1, -2)
        return outputs.reshape(-1, outputs.shape[-1]) * self.scalings.to(outputs.dtype)

    def run_qkv_lora(
        self,
        x: torch.Tensor,
        qkv_lora_a: torch.Tensor,
        qkv_lora_b: Union[torch.Tensor, Tuple[torch.Tensor]],
        *args,
        **kwargs,
    ) -> torch.Tensor:
        """Run the lora pass for QKV Layer.

        Args:
            x: input matrix with shape (s, input_dim), here s is the sum of all sequence lengths
            qkv_lora_a: lora_a module for qkv, with shape (num_lora, 3 * r, input_dim)
            qkv_lora_b: lora_b module for qkv.
                        If passed in as a tensor, its shape should be (num_lora,output_dim_q + 2 * output_dim_kv, r)
                        If passed in as a tuple of two tensors, it should contain:
                           a lora_b module for q, with shape (1, num_lora, output_dim_q, r)
                           and a combined lora_b module for kv, with shape (2, num_lora, output_dim_kv, r)
        Returns:
            result with shape (s, output_dim_q + 2 * output_dim_kv)
        """
        assert isinstance(qkv_lora_b, tuple) and len(qkv_lora_b) == 2

        # Shape of lora_a_output: (s, 3 * r)
        lora_a_output = self.run_lora_a_sgemm(x=x, weights=qkv_lora_a)

        q_lora_b, kv_lora_b = qkv_lora_b
        lora_rank = kv_lora_b.shape[-1]
        output_dim_q = q_lora_b.shape[-2]
        output_dim_kv = kv_lora_b.shape[-2]
        lora_output = torch.empty(
            (x.shape[0], output_dim_q + 2 * output_dim_kv),
            device=x.device,
            dtype=x.dtype,
        )

        # q
        lora_output[:, :output_dim_q] = self.run_lora_b_sgemm(
            x=lora_a_output[:, :lora_rank].contiguous(), weights=q_lora_b[0]
        )

        # kv
        lora_output[:, output_dim_q : output_dim_q + output_dim_kv] = (
            self.run_lora_b_sgemm(
                x=lora_a_output[:, lora_rank : 2 * lora_rank].contiguous(),
                weights=kv_lora_b[0],
            )
        )

        lora_output[
            :, output_dim_q + output_dim_kv : output_dim_q + 2 * output_dim_kv
        ] = self.run_lora_b_sgemm(
            x=lora_a_output[:, 2 * lora_rank : 3 * lora_rank].contiguous(),
            weights=kv_lora_b[1],
        )

        return lora_output

    def run_gate_up_lora(
        self,
        x: torch.Tensor,
        gate_up_lora_a: torch.Tensor,
        gate_up_lora_b: Union[torch.Tensor, Tuple[torch.Tensor]],
        *args,
        **kwargs,
    ) -> torch.Tensor:
        """Run the lora pass for gate_up_proj, usually attached to MergedColumnParallelLayer.

        Args:
            x: input matrix with shape (s, input_dim), here s is the sum of all sequence lengths
            gate_up_lora_a: lora_a module for gate_up_proj, with shape (num_lora, 2 * r, input_dim)
            gate_up_lora_b: lora_b module for qkv.
                        If passed in as a tensor, its shape should be (num_lora, 2 * output_dim, r)
                        If passed in as a tuple, it should contain two tensors with shape (num_lora, output_dim, r)
        Returns:
            result with shape (s, 2 * output_dim)
        """
        assert isinstance(gate_up_lora_b, tuple) and len(gate_up_lora_b) == 2
        lora_rank = gate_up_lora_b[0].shape[-1]
        output_dim = gate_up_lora_b[0].shape[-2]

        # Shape of lora_a_output: (s, 2 * r)
        lora_a_output = self.run_lora_a_sgemm(x=x, weights=gate_up_lora_a)

        lora_output = torch.empty(
            (x.shape[0], 2 * output_dim),
            device=x.device,
            dtype=x.dtype,
        )

        # Compute lora for gate and up proj respectively
        lora_output[:, :output_dim] = self.run_lora_b_sgemm(
            x=lora_a_output[:, :lora_rank].contiguous(),
            weights=gate_up_lora_b[0],
        )

        lora_output[:, output_dim:] = self.run_lora_b_sgemm(
            x=lora_a_output[:, lora_rank:].contiguous(),
            weights=gate_up_lora_b[1],
        )

        return lora_output

    def set_batch_info(self, batch_info: LoRABatchInfo):
        self.batch_info = batch_info
        self.exploded_indices = torch.repeat_interleave(
            self.batch_info.weight_indices, self.batch_info.seg_lens.to(torch.int32)
        )
        self.scalings = torch.gather(
            self.batch_info.scalings, 0, self.exploded_indices
        ).unsqueeze(-1)
