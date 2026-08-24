from __future__ import annotations

import functools

import torch
import torch.distributed as dist
from vllm.config import CUDAGraphMode
from vllm.distributed.parallel_state import get_dp_group
from vllm_ascend.ascend_forward_context import select_moe_comm_method
from vllm_ascend.ops.fused_moe.moe_comm_method import MoECommType
from vllm_ascend.utils import (
    is_moe_model,
    should_skip_allreduce_across_dp_group,
)


_PATCH_APPLIED = False
_PATCH_MARKER = "_modelarts_moe_dp_metadata_sync_applied"
_NO_FORWARD_PATCH_MARKER = "_modelarts_empty_batch_dp_sync_applied"


def apply_patch() -> None:
    """Keep MoE communication and ACLGraph modes consistent across DP ranks."""
    global _PATCH_APPLIED

    if _PATCH_APPLIED:
        return

    from vllm_ascend.worker.model_runner_v1 import NPUModelRunner

    original_kv_connector_no_forward = (
        NPUModelRunner.kv_connector_no_forward
    )

    if not getattr(
        original_kv_connector_no_forward,
        _NO_FORWARD_PATCH_MARKER,
        False,
    ):

        @functools.wraps(original_kv_connector_no_forward)
        def kv_connector_no_forward_with_dp_sync(
            self,
            scheduler_output,
            *args,
            **kwargs,
        ):
            if (
                scheduler_output.total_num_scheduled_tokens > 0
                and self.parallel_config.distributed_executor_backend
                == "external_launcher"
                and self.parallel_config.data_parallel_size > 1
            ):
                self._dummy_run(1)

            return original_kv_connector_no_forward(
                scheduler_output,
                *args,
                **kwargs,
            )

        setattr(
            kv_connector_no_forward_with_dp_sync,
            _NO_FORWARD_PATCH_MARKER,
            True,
        )
        NPUModelRunner.kv_connector_no_forward = (
            kv_connector_no_forward_with_dp_sync
        )

    original_sync_metadata_across_dp = NPUModelRunner._sync_metadata_across_dp

    if getattr(
        original_sync_metadata_across_dp,
        _PATCH_MARKER,
        False,
    ):
        _PATCH_APPLIED = True
        return

    @functools.wraps(original_sync_metadata_across_dp)
    def sync_metadata_across_dp_with_moe_consistency(
        self,
        num_tokens: int,
        is_draft_model: bool = False,
        cudagraph_mode: CUDAGraphMode = CUDAGraphMode.NONE,
        allow_dp_padding: bool = False,
    ) -> tuple[int, torch.Tensor | None, CUDAGraphMode]:
        needs_moe_metadata_sync = (
            self.dp_size > 1
            and is_moe_model(self.vllm_config)
            and should_skip_allreduce_across_dp_group(
                self.vllm_config,
                is_draft_model,
            )
        )

        if not needs_moe_metadata_sync:
            return original_sync_metadata_across_dp(
                self,
                num_tokens,
                is_draft_model,
                cudagraph_mode,
                allow_dp_padding,
            )

        # Each rank writes its metadata into its own column. After SUM
        # all-reduce, every rank obtains the complete DP metadata.
        dp_metadata = torch.zeros(
            (2, self.dp_size),
            device="cpu",
            dtype=torch.int32,
        )
        dp_metadata[0, self.dp_rank] = num_tokens
        dp_metadata[1, self.dp_rank] = cudagraph_mode.value

        dist.all_reduce(
            dp_metadata,
            group=get_dp_group().cpu_group,
        )

        tokens_across_dp = dp_metadata[0]
        max_tokens_across_dp = int(tokens_across_dp.max().item())

        # CUDAGraphMode: NONE=0, PIECEWISE=1, FULL=2.
        # Use the most conservative mode supported by every rank.
        synced_cudagraph_mode = CUDAGraphMode(int(dp_metadata[1].min().item()))

        comm_methods = set()

        for rank_tokens in tokens_across_dp:
            comm_method = select_moe_comm_method(int(rank_tokens.item()), self.vllm_config)
            comm_methods.add(comm_method)

        uneven_token_methods = {
            MoECommType.MC2,
            MoECommType.FUSED_MC2,
        }

        can_keep_uneven_tokens = (
            len(comm_methods) == 1
            and comm_methods.issubset(uneven_token_methods)
        )

        if can_keep_uneven_tokens:
            # All ranks use the same communication method, which supports
            # different local token counts.
            local_tokens_for_dp = torch.full(
                (self.dp_size,),
                num_tokens,
                device="cpu",
                dtype=torch.int32,
            )
            return (
                num_tokens,
                local_tokens_for_dp,
                synced_cudagraph_mode,
            )

        # Communication methods may diverge. Pad every rank to the global
        # maximum so all ranks select the same method.
        uniform_tokens_for_dp = torch.full(
            (self.dp_size,),
            max_tokens_across_dp,
            device="cpu",
            dtype=torch.int32,
        )
        return (
            max_tokens_across_dp,
            uniform_tokens_for_dp,
            synced_cudagraph_mode,
        )

    setattr(
        sync_metadata_across_dp_with_moe_consistency,
        _PATCH_MARKER,
        True,
    )

    NPUModelRunner._sync_metadata_across_dp = sync_metadata_across_dp_with_moe_consistency

    _PATCH_APPLIED = True


apply_patch()
