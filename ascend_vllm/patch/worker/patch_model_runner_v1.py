from __future__ import annotations

import functools
from copy import copy
from typing import TYPE_CHECKING, Any

import numpy as np
import torch
import torch.distributed as dist
from torch import nn
from vllm.config import CUDAGraphMode
from vllm.distributed.parallel_state import get_dp_group
from vllm.v1.attention.backends.gdn_attn import GDNAttentionMetadataBuilder
from vllm.v1.attention.backends.utils import CommonAttentionMetadata
from vllm.v1.core.sched.output import SchedulerOutput
from vllm.v1.kv_cache_interface import EncoderOnlyAttentionSpec
from vllm.v1.spec_decode.metadata import SpecDecodeMetadata
from vllm.v1.worker.ubatch_utils import UBatchSlices

import vllm_ascend.envs as envs_ascend
from vllm_ascend.attention.attention_v1 import AscendAttentionState
from vllm_ascend.attention.context_parallel.dsa_cp import (
    AscendDSACPMetadataBuilder,
)
from vllm_ascend.attention.context_parallel.sfa_cp import (
    AscendSFADCPMetadataBuilder,
)
from vllm_ascend.attention.dsa_v1 import AscendDSAMetadataBuilder
from vllm_ascend.attention.utils import AscendCommonAttentionMetadata
from vllm_ascend.spec_decode.dflash_proposer import AscendDflashProposer
from vllm_ascend.spec_decode.draft_proposer import AscendDraftModelProposer
from vllm_ascend.spec_decode.dspark_proposer import AscendDSparkProposer
from vllm_ascend.spec_decode.eagle_proposer import AscendEagleProposer
from vllm_ascend.spec_decode.step3p5 import AscendStep3p5MTPProposer
from vllm_ascend.spec_decode.utils import (
    update_num_computed_tokens_for_batch_change,
)
from vllm_ascend.utils import (
    is_moe_model,
    lmhead_tp_enable,
    should_skip_allreduce_across_dp_group,
)
from vllm_ascend.worker.dcp_utils import DCPAsyncSpecDecodeRebuildResult

if TYPE_CHECKING:
    from vllm_ascend.worker.model_runner_v1 import PerLayerAttnMetadata

_PATCH_APPLIED = False
_PATCH_MARKER = "_modelarts_moe_dp_metadata_sync_applied"

def _prepare_inputs(
    self,
    scheduler_output: "SchedulerOutput",
    num_scheduled_tokens: np.ndarray,
) -> tuple[
    torch.Tensor,
    SpecDecodeMetadata | None,
    int,
]:
    """
    :return: tuple[
        logits_indices,
        spec_decode_metadata,
        total_num_scheduled_tokens,
    ]
    """
    total_num_scheduled_tokens = scheduler_output.total_num_scheduled_tokens
    assert total_num_scheduled_tokens > 0
    num_reqs = self.input_batch.num_reqs
    assert num_reqs > 0

    # OPTIMIZATION: Start copying the block table first.
    # This way, we can overlap the copy with the following CPU operations.
    self.input_batch.block_table.commit_block_table(num_reqs)

    req_indices = np.repeat(self.arange_np[:num_reqs], num_scheduled_tokens)

    # Get the attention state.
    if not scheduler_output.scheduled_spec_decode_tokens:
        num_valid_tokens = num_scheduled_tokens
    else:
        num_valid_tokens = np.array(
            [
                scheduler_output.num_scheduled_tokens[i]
                - len(scheduler_output.scheduled_spec_decode_tokens.get(i, []))
                for i in self.input_batch.req_ids
            ],
            dtype=np.int32,
        )
    attn_state = self._build_attn_state(num_reqs, num_scheduled_tokens, num_valid_tokens)

    # Determine if it's a splitfuse batch
    with_prefill = attn_state not in [AscendAttentionState.DecodeOnly, AscendAttentionState.SpecDecoding]
    self.with_prefill = with_prefill

    # Get positions.
    cu_num_tokens = self._get_cumsum_and_arange(
        num_scheduled_tokens, self.query_pos.np
    )
    positions_np = self._positions_np_buf[:total_num_scheduled_tokens]
    np.add(
        self.input_batch.num_computed_tokens_cpu[req_indices],
        self.query_pos.np[: cu_num_tokens[-1]],
        out=positions_np,
    )

    if self.use_dcp:
        self.dcp_manager.init_batch_info(
            num_scheduled_tokens,
            self.input_batch.num_reqs,
            self.input_batch.num_computed_tokens_cpu,
            self.input_batch.num_prompt_tokens,
        )

    # Build previous positions before DCP prepares speculative inputs.
    prev_req_id_to_index = self.input_batch.prev_req_id_to_index
    self._compute_prev_positions(num_reqs)
    prev_positions_gpu = None
    if (
        self.use_async_scheduling
        and self.input_batch.prev_sampled_token_ids is not None
        and prev_req_id_to_index
    ):
        self.prev_positions.copy_to_gpu(num_reqs)
        prev_positions_gpu = self.prev_positions.gpu[:num_reqs]

    if self.speculative_config and self.use_dcp:
        self.dcp_manager.generate_dcp_mtp_input(
            total_num_scheduled_tokens,
            scheduler_output.num_scheduled_tokens,
            with_prefill,
            self.input_batch,
            self.arange_np,
            req_indices,
            positions_np,
            cu_num_tokens,
            self._draft_token_ids,  # type: ignore[has-type]
            scheduler_output,
            self.num_spec_tokens,
            prev_positions=prev_positions_gpu,
        )

    self.query_lens = torch.from_numpy(num_scheduled_tokens)

    # Get token indices.
    # E.g., [0, 1, 0, 1, 2, 3, 4, 0, 1, 2]
    # -> [0, 1, M, M + 1, M + 2, M + 3, M + 4, 2 * M, 2 * M + 1, 2 * M + 2]
    # where M is the max_model_len.
    token_indices = positions_np + req_indices * self.input_batch.token_ids_cpu.shape[1]
    token_indices_tensor = torch.from_numpy(token_indices)
    # Prepare input_ids.
    # NOTE(woosuk): We use torch.index_select instead of np.take here
    # because torch.index_select is much faster than np.take for large
    # tensors.
    torch.index_select(
        self.input_batch.token_ids_cpu_tensor.flatten(),
        0,
        token_indices_tensor,
        out=self.input_ids.cpu[:total_num_scheduled_tokens],
    )
    if self.enable_prompt_embeds:
        is_token_ids = self.input_batch.is_token_ids_tensor.flatten()
        torch.index_select(
            is_token_ids, 0, token_indices_tensor, out=self.is_token_ids.cpu[:total_num_scheduled_tokens]
        )

    # Because we did not pre-allocate a massive prompt_embeds CPU tensor on
    # the InputBatch, we need to fill in the prompt embeds into the expected
    # spots in the GpuModelRunner's pre-allocated prompt_embeds tensor.
    if self.input_batch.req_prompt_embeds and (self.is_multimodal_model or self.enable_prompt_embeds):
        output_idx = 0
        for req_idx in range(num_reqs):
            num_sched = num_scheduled_tokens[req_idx]

            # Skip if this request doesn't have embeddings
            if req_idx not in self.input_batch.req_prompt_embeds:
                output_idx += num_sched
                continue

            # Skip if no tokens scheduled
            if num_sched <= 0:
                output_idx += num_sched
                continue

            req_embeds = self.input_batch.req_prompt_embeds[req_idx]
            start_pos = self.input_batch.num_computed_tokens_cpu[req_idx]

            # Skip if trying to read beyond available embeddings
            if start_pos >= req_embeds.shape[0]:
                output_idx += num_sched
                continue

            # Copy available embeddings
            end_pos = start_pos + num_sched
            actual_end = min(end_pos, req_embeds.shape[0])
            actual_num_sched = actual_end - start_pos

            if actual_num_sched > 0:
                self.inputs_embeds.cpu[output_idx : output_idx + actual_num_sched].copy_(
                    req_embeds[start_pos:actual_end]
                )

            output_idx += num_sched

    self.query_start_loc.np[0] = 0
    self.query_start_loc.np[1 : num_reqs + 1] = cu_num_tokens
    self.query_start_loc.copy_to_gpu()

    # Now, query_start_loc is padded.
    # But gdn needs an unpadded one.
    # gdn_query_start_loc is an unpadded version of query_start_loc.
    # TODO delete it if fia's check is removed.
    if self._has_gdn:
        self.gdn_query_start_loc.np[0] = 0
        self.gdn_query_start_loc.np[1 : num_reqs + 1] = cu_num_tokens
        self.gdn_query_start_loc.np[num_reqs + 1 :].fill(cu_num_tokens[-1])
        self.gdn_query_start_loc.copy_to_gpu()


    # Compute optimistic seq_lens (assumes all draft tokens from previous
    # iteration accepted). Store in optimistic_seq_lens_cpu for use by
    # _build_attention_metadata (max_seq_len) and discard_request_mask.
    # seq_lens (GPU) will be computed later using the same optimistic values.
    torch.add(
        self.input_batch.num_computed_tokens_cpu_tensor[:num_reqs],
        torch.from_numpy(num_scheduled_tokens),
        out=self.optimistic_seq_lens_cpu[:num_reqs],
    )
    self.optimistic_seq_lens_cpu[num_reqs:].fill_(0)

    # Fill unused with -1. Needed for reshape_and_cache in attention_cp
    self.query_start_loc.gpu[num_reqs + 1 :].fill_(-1)

    # Copy the tensors to the NPU.
    self._prepare_input_ids(scheduler_output, num_reqs, total_num_scheduled_tokens, cu_num_tokens)
    # Calculate M-RoPE positions.
    # Only relevant for models using M-RoPE (e.g, Qwen2-VL)
    if self.uses_mrope:
        # Only relevant for models using M-RoPE (e.g, Qwen2-VL)
        self._calc_mrope_positions(scheduler_output)
        self.mrope_positions.gpu.copy_(
            self.mrope_positions.cpu,
            non_blocking=True,
        )
    elif self.uses_xdrope_dim > 0:
        self._calc_xdrope_positions(scheduler_output)
        # Only relevant for models using XD-RoPE (e.g, HunYuan-VL)
        self.xdrope_positions.gpu[:, :total_num_scheduled_tokens].copy_(
            self.xdrope_positions.cpu[:, :total_num_scheduled_tokens],
            non_blocking=True,
        )

    # Record the index of requests that should not be sampled,
    # so that we could clear the sampled tokens before returning
    num_tokens = [self.requests[r].num_tokens for r in self.input_batch.req_ids]
    num_tokens_np = np.array(num_tokens, dtype=np.int32)
    base_num_reqs = self.input_batch.num_reqs
    num_reqs = base_num_reqs
    discard_requests_mask = self.optimistic_seq_lens_cpu[:num_reqs].numpy() < num_tokens_np

    discard_request_indices = np.nonzero(discard_requests_mask)[0]
    self.num_discarded_requests = len(discard_request_indices)
    self.discard_request_indices.np[: self.num_discarded_requests] = discard_request_indices
    self.discard_request_indices.copy_to_gpu(self.num_discarded_requests)
    
    self.discard_request_mask.np[:num_reqs] = discard_requests_mask
    self.discard_request_mask.copy_to_gpu(num_reqs)

    # Sync num_accepted_tokens from CPU (set by
    # _update_states_after_model_execute for hybrid models).
    if self.num_accepted_tokens_event is not None:
        # ascend vllm adaptor start: zero-bubble fast path
        if (
            self.use_async_scheduling
            and self.need_accepted_tokens
            and prev_req_id_to_index
            and self.cache_config.mamba_cache_mode == "none"
            and envs_ascend.ENABLE_ZERO_BUBBLE
        ):
            # Fast path: fix up num_accepted_tokens entirely on the NPU
            # without blocking the CPU on num_accepted_tokens_event.
            torch.npu.current_stream().wait_event(self.num_accepted_tokens_event)
            if prev_positions_gpu is None:
                self.prev_positions.copy_to_gpu(num_reqs)
                prev_positions_gpu = self.prev_positions.gpu[:num_reqs]
            num_accepted_tokens_gpu = self.num_accepted_tokens.gpu
            gathered = num_accepted_tokens_gpu.gather(
                0, prev_positions_gpu.clamp(min=0)
            )
            num_accepted_tokens_gpu[:num_reqs].copy_(
                torch.where(prev_positions_gpu < 0, 1, gathered)
            )
            num_accepted_tokens_gpu[num_reqs:].fill_(1)
        else:
            self.num_accepted_tokens_event.synchronize()
            # Async mode: condense() reordered indices, use prev_positions mapping
            if self.use_async_scheduling and prev_req_id_to_index:
                prev_idx = self.prev_positions.np[:num_reqs]
                new_mask = prev_idx < 0
                self.num_accepted_tokens.np[:num_reqs] = (
                    self.input_batch.num_accepted_tokens_cpu[
                        np.where(new_mask, 0, prev_idx)
                    ]
                )
                self.num_accepted_tokens.np[:num_reqs][new_mask] = 1
                self.input_batch.num_accepted_tokens_cpu[:num_reqs] = (
                    self.num_accepted_tokens.np[:num_reqs]
                )
            else:
                # Non-async mode: use values directly
                self.num_accepted_tokens.np[:num_reqs] = (
                    self.input_batch.num_accepted_tokens_cpu[:num_reqs]
                )

            # Both CPU fallback branches must upload the updated counts.
            self.num_accepted_tokens.np[num_reqs:].fill(1)
            self.num_accepted_tokens.copy_to_gpu()
        # ascend vllm adaptor end    
    else:
        self.num_accepted_tokens.np.fill(1)
        self.num_accepted_tokens.gpu.fill_(1)

    # Update num_computed_tokens on GPU. In async spec decode,
    # CPU values are optimistic (all drafts accepted). The kernel
    # corrects on GPU using the previous step's
    # valid_sampled_token_count_gpu. Otherwise, just copy from CPU.
    valid_sampled_token_count_gpu = self.valid_sampled_token_count_gpu
    if self.use_async_spec_decode:
        computed_token_tensor_cpu = self.input_batch.num_computed_tokens_cpu_tensor[:num_reqs].to(
            device=self.device, non_blocking=True
        )
    if (
        self.use_async_spec_decode
        and valid_sampled_token_count_gpu is not None
        and prev_req_id_to_index
    ):
        if prev_positions_gpu is None:
            self.prev_positions.copy_to_gpu(num_reqs)
        self.prev_num_draft_tokens.copy_to_gpu()
        update_num_computed_tokens_for_batch_change(
            self.num_computed_tokens,
            self.num_accepted_tokens.gpu[:num_reqs],
            self.prev_positions.gpu[:num_reqs],
            valid_sampled_token_count_gpu,
            self.prev_num_draft_tokens.gpu,
            computed_token_tensor_cpu,
        )
    else:
        self.num_computed_tokens[:num_reqs].copy_(
            self.input_batch.num_computed_tokens_cpu_tensor[:num_reqs],
            non_blocking=True,
        )

    self.req_indices.np[:total_num_scheduled_tokens] = req_indices
    self.req_indices.copy_to_gpu(total_num_scheduled_tokens)
    req_indices_gpu = self.req_indices.gpu[:total_num_scheduled_tokens]

    self.query_pos.copy_to_gpu(total_num_scheduled_tokens)
    self.num_scheduled_tokens.np[:num_reqs] = num_scheduled_tokens
    self.num_scheduled_tokens.copy_to_gpu(num_reqs)
    num_scheduled_tokens_gpu = self.num_scheduled_tokens.gpu[:num_reqs]

    dcp_manager = getattr(self, "dcp_manager", None)
    if dcp_manager is not None:
        cp_async_rebuild = dcp_manager.rebuild_async_spec_decode_inputs(
            use_async_spec_decode=self.use_async_spec_decode,
            valid_sampled_token_count_gpu=valid_sampled_token_count_gpu,
            prev_req_id_to_index=prev_req_id_to_index,
            prev_positions_gpu=prev_positions_gpu,
            with_prefill=with_prefill,
            enable_prompt_embeds=self.enable_prompt_embeds,
            has_req_prompt_embeds=bool(self.input_batch.req_prompt_embeds),
            supports_mm_inputs=self.supports_mm_inputs,
            num_reqs=num_reqs,
            total_num_scheduled_tokens=total_num_scheduled_tokens,
            req_indices=req_indices,
            req_indices_gpu=req_indices_gpu,
            query_pos_gpu=self.query_pos.gpu,
            query_pos_np=self.query_pos.np,
            positions=self.positions,
            positions_np=positions_np,
            num_computed_tokens=self.num_computed_tokens,
            num_computed_tokens_cpu=self.input_batch.num_computed_tokens_cpu,
            prev_positions_np=self.prev_positions.np,
            prev_num_draft_tokens_np=self.prev_num_draft_tokens.np,
            valid_sampled_token_count_event=self.valid_sampled_token_count_event,
            valid_sampled_token_count_cpu=self.valid_sampled_token_count_cpu,
            input_batch=self.input_batch,
            input_ids=self.input_ids,
            scheduler_output=scheduler_output,
            arange_np=self.arange_np,
            cu_num_tokens=cu_num_tokens,
            draft_token_ids=self._draft_token_ids,  # type: ignore[has-type]
            num_spec_tokens=self.num_spec_tokens,
            prepare_input_ids=self._prepare_input_ids,
        )
    else:
        cp_async_rebuild = DCPAsyncSpecDecodeRebuildResult(
            rebuilt=False,
            positions_ready_on_device=False,
        )

    if cp_async_rebuild.positions_ready_on_device:
        pass
    elif cp_async_rebuild.rebuilt:
        # The async rebuild computed corrected positions on CPU.
        # Copy positions_np to GPU so input_ids and positions stay aligned.

        self.positions[:total_num_scheduled_tokens].copy_(
            torch.from_numpy(
                positions_np[:total_num_scheduled_tokens]
            ).to(self.device),
            non_blocking=True,
        )
    else:
        self.positions[:total_num_scheduled_tokens] = (
            self.num_computed_tokens[req_indices_gpu].to(torch.int64)
            + self.query_pos.gpu[:total_num_scheduled_tokens]
        )

    self.seq_lens[:num_reqs] = (
        self.num_computed_tokens[:num_reqs] + num_scheduled_tokens_gpu
    )
    self.seq_lens[num_reqs:].fill_(0)

    # In async spec decode mode, optimistic_seq_lens_cpu assumes all
    # tokens from the previous speculative step were accepted. Correct it
    # on CPU using the valid-sampled-token counts that are already copied
    # asynchronously for scheduler bookkeeping. This avoids an extra
    # NPU->CPU seq_lens copy and the synchronize() in attention metadata.
    # Mirrors update_num_computed_tokens_for_batch_change on the GPU side.
    async_spec_decode_active = (
        self.use_async_spec_decode
        and valid_sampled_token_count_gpu is not None
        and prev_req_id_to_index
    )
    # ascend vllm adaptor start: zero-bubble deferred seq_lens correction
    needs_correction = bool(
        self._needs_seq_lens_cpu_sync and async_spec_decode_active
    )
    self._seq_lens_cpu_correction_pending = bool(
        envs_ascend.ENABLE_ZERO_BUBBLE and needs_correction
    )

    if needs_correction and not envs_ascend.ENABLE_ZERO_BUBBLE:
        self._correct_optimistic_seq_lens_cpu(num_reqs)
    # ascend vllm adaptor end

    self.input_batch.block_table.compute_slot_mapping(
        num_reqs,
        self.query_start_loc.gpu[: num_reqs + 1],
        self.positions[:total_num_scheduled_tokens],
    )

    if self.use_async_spec_decode and (self.uses_mrope or self.uses_xdrope_dim > 0):
        drift = self.num_computed_tokens[req_indices_gpu].to(
            torch.int64
        ) - computed_token_tensor_cpu[req_indices_gpu]
        target = self.mrope_positions if self.uses_mrope else self.xdrope_positions
        target.gpu[:, :total_num_scheduled_tokens] += drift

    use_spec_decode = len(scheduler_output.scheduled_spec_decode_tokens) > 0
    if not use_spec_decode:
        # NOTE(woosuk): Due to chunked prefills, the batch may contain
        # partial requests. While we should not sample any token
        # from these partial requests, we do so for simplicity.
        # We will ignore the sampled tokens from the partial requests.
        # TODO: Support prompt logprobs.
        spec_decode_metadata = None
        num_draft_tokens = None
        num_sampled_tokens = np.ones(num_reqs, dtype=np.int32)
        logits_indices = self.query_start_loc.gpu[1 : num_reqs + 1] - 1
    else:
        # Get the number of draft tokens for each request.
        # Iterate over the dictionary rather than all requests since not all
        # requests have draft tokens.
        num_draft_tokens = np.zeros(num_reqs, dtype=np.int32)
        # For chunked prefills, use -1 as mask rather than 0, as guided
        # decoding may rollback speculative tokens.
        new_schedule_reqs = [x.req_id for x in scheduler_output.scheduled_new_reqs]
        num_decode_draft_tokens = np.full(num_reqs, -1, dtype=np.int32)
        for (
            req_id,
            draft_token_ids,
        ) in scheduler_output.scheduled_spec_decode_tokens.items():
            req_idx = self.input_batch.req_id_to_index[req_id]
            draft_len = len(draft_token_ids)
            num_draft_tokens[req_idx] = draft_len
            if (self.is_kv_consumer and req_id in new_schedule_reqs) or \
                (self.input_batch.num_computed_tokens_cpu[req_idx] >= \
                self.input_batch.num_prompt_tokens[req_idx]):
                num_decode_draft_tokens[req_idx] = draft_len
            else:
                num_decode_draft_tokens[req_idx] = -1

        spec_decode_metadata = self._calc_spec_decode_metadata(
            num_draft_tokens,
            cu_num_tokens,
        )
        logits_indices = spec_decode_metadata.logits_indices
        num_sampled_tokens = num_draft_tokens + 1

        # For DECODE only cuda graph of some attention backends (e.g., GDN).
        self.num_decode_draft_tokens.np[:num_reqs] = num_decode_draft_tokens
        self.num_decode_draft_tokens.np[num_reqs:].fill(-1)
        self.num_decode_draft_tokens.copy_to_gpu()
    self.logits_indices = logits_indices

    # Hot-Swap lora model
    if self.lora_config:
        assert np.sum(num_sampled_tokens) <= self.vllm_config.scheduler_config.max_num_batched_tokens
        self.set_active_loras(self.input_batch, num_scheduled_tokens, num_sampled_tokens)
    if lmhead_tp_enable():
        max_num_reqs_across_dp = self.max_num_reqs * self.uniform_decode_query_len
        logits_indices = nn.functional.pad(logits_indices, (0, max_num_reqs_across_dp - logits_indices.shape[0]))

    return (
        logits_indices,
        spec_decode_metadata,
        total_num_scheduled_tokens,
    )


def _build_attention_metadata(
    self,
    num_tokens: int,
    num_reqs: int,
    max_query_len: int,
    num_tokens_padded: int | None = None,
    num_reqs_padded: int | None = None,
    ubatch_slices: UBatchSlices | None = None,
    logits_indices: torch.Tensor | None = None,
    use_spec_decode: bool = False,
    for_cudagraph_capture: bool = False,
    num_scheduled_tokens: dict[str, int] | None = None,
    num_scheduled_tokens_np: np.ndarray | None = None,
    cascade_attn_prefix_lens: list[list[int]] | None = None,
) -> tuple[PerLayerAttnMetadata, CommonAttentionMetadata | None]:
    """
    :return: tuple[attn_metadata, spec_decode_common_attn_metadata]
    """
    # Attention metadata is not needed for attention free models
    if len(self.kv_cache_config.kv_cache_groups) == 0:
        return {}, None
    num_tokens_padded = num_tokens_padded or num_tokens
    num_reqs_padded = num_reqs_padded or num_reqs
    attn_metadata: PerLayerAttnMetadata = {}
    if ubatch_slices is not None:
        attn_metadata = [dict() for _ in range(len(ubatch_slices))]

    # ascend vllm adaptor start
    # Capture uses dummy inputs; do not consume a real batch's pending flag.
    correction_pending = bool(
        envs_ascend.ENABLE_ZERO_BUBBLE
        and not for_cudagraph_capture
        and getattr(self, "_seq_lens_cpu_correction_pending", False)
    )
    correction_applied = False

    def correct_seq_lens_cpu_if_pending() -> None:
        nonlocal correction_pending, correction_applied

        if not correction_pending:
            return

        self._correct_optimistic_seq_lens_cpu(num_reqs)
        self._seq_lens_cpu_correction_pending = False
        correction_pending = False
        correction_applied = True

    # DCP reads CPU lengths before the attention-group loop.
    if self.use_dcp:
        correct_seq_lens_cpu_if_pending()
    # ascend vllm adaptor end

    if for_cudagraph_capture:
        # For some attention backends (e.g. FA) with sliding window models we need
        # to make sure the backend see a max_seq_len that is larger to the sliding
        # window size when capturing to make sure the correct kernel is selected.
        max_seq_len = self.max_model_len
    else:
        max_seq_len = self.optimistic_seq_lens_cpu.numpy()[:num_reqs].max().item()


    kv_cache_groups = self.kv_cache_config.kv_cache_groups

    def _get_dcp_metadata(block_table_tensor):
        if not self.use_dcp:
            return None, block_table_tensor

        fixed_decode_seq_lens_cpu = None
        if self.use_async_spec_decode:
            fixed_decode_seq_lens_cpu = self.optimistic_seq_lens_cpu[:num_reqs].numpy()

        assert num_reqs_padded is not None
        return self.dcp_manager.generate_dcp_metadata(
            num_tokens,
            self.query_lens,
            self.input_batch,
            num_scheduled_tokens_np,
            block_table_tensor,
            num_reqs_padded,
            num_reqs,
            fixed_decode_seq_lens_cpu,
        )

    def _get_block_table_and_slot_mapping(
        kv_cache_gid: int,
    ):
        assert num_reqs_padded is not None and num_tokens_padded is not None
        kv_cache_spec = kv_cache_groups[kv_cache_gid].kv_cache_spec
        if isinstance(kv_cache_spec, EncoderOnlyAttentionSpec):
            blk_table_tensor = torch.zeros(
                (num_reqs_padded, 1),
                dtype=torch.int32,
                device=self.device,
            )
            slot_mapping = torch.zeros(
                (num_tokens_padded,),
                dtype=torch.int64,
                device=self.device,
            )
        else:
            blk_table = self.input_batch.block_table[kv_cache_gid]
            slot_mapping = blk_table.slot_mapping.gpu[:num_tokens_padded]
            blk_table_tensor = blk_table.get_device_tensor()[:num_reqs_padded]
            # Fill unused with -1. Needed for reshape_and_cache in full cuda
            # graph mode. `blk_table_tensor` -1 to match mamba PAD_SLOT_ID
            slot_mapping[num_tokens:num_tokens_padded].fill_(-1)
            blk_table_tensor[num_reqs:num_reqs_padded].fill_(0)
        if self.model_config.enable_return_routed_experts and kv_cache_gid == 0:
            if self.routed_experts_initialized:
                # snapshot slot_mapping into a private device
                # buffer so the next ``_prepare_inputs`` does not
                # overwrite it while D2H is still pending.
                n = slot_mapping.shape[0]
                self.routed_experts_slot_mapping_device[:n].copy_(
                    slot_mapping
                )
        return blk_table_tensor, slot_mapping

    block_table_gid_0, slot_mapping_gid_0 = _get_block_table_and_slot_mapping(0)
    self.long_seq_metadata, block_table_gid_0 = _get_dcp_metadata(block_table_gid_0)
    num_computed_tokens_cpu = self.input_batch.num_computed_tokens_cpu_tensor[
        :num_reqs_padded
    ]
    num_prompt_tokens_cpu = self.input_batch.num_prompt_tokens_cpu_tensor[
        :num_reqs_padded
    ]
    is_prefilling = num_computed_tokens_cpu < num_prompt_tokens_cpu
    is_prefilling[num_reqs:] = False
    seq_lens_cpu = self.optimistic_seq_lens_cpu[:num_reqs_padded]
    if self.use_async_spec_decode:
        # GPU tensors are authoritative in async mode.
        seq_lens_cpu = None
        num_computed_tokens_cpu = None

    cm_base = AscendCommonAttentionMetadata(
        query_start_loc=self.query_start_loc.gpu[: num_reqs_padded + 1],
        query_start_loc_cpu=self.query_start_loc.cpu[: num_reqs_padded + 1],
        seq_lens=self.seq_lens[:num_reqs_padded],
        # Always pass optimistic_seq_lens_cpu via _seq_lens_cpu so NPU
        # attention backends can get CPU seq_lens without GPU->CPU sync.
        # This is separate from seq_lens_cpu (None in async) which eagle
        # proposer checks to distinguish async/non-async behavior.
        _seq_lens_cpu=self.optimistic_seq_lens_cpu[:num_reqs_padded],
        seq_lens_cpu_upper_bound=self.optimistic_seq_lens_cpu[:num_reqs_padded],
        # TODO
        seq_lens_cpu=seq_lens_cpu,
        # TODO
        # num_computed_tokens_cpu=self.input_batch.num_computed_tokens_cpu_tensor[:num_reqs_padded],
        num_computed_tokens_cpu=num_computed_tokens_cpu,
        num_reqs=num_reqs_padded,
        num_actual_tokens=num_tokens,
        max_query_len=max_query_len,
        max_seq_len=max_seq_len,
        block_table_tensor=block_table_gid_0,
        slot_mapping=slot_mapping_gid_0,
        causal=True,
        is_prefilling=is_prefilling,
        num_input_tokens=num_tokens_padded,
        actual_seq_lengths_q=self.actual_seq_lengths_q,
        positions=self.positions,
        positions_cpu=self._dsa_positions_cpu_buf if self.use_compress else None,
        attn_state=self.attn_state,
        decode_token_per_req=self.decode_token_per_req,
        context_parallel_metadata=self.long_seq_metadata,
        group_len = self.group_len.gpu[:num_reqs_padded],
        group_key_idx = self.group_key_idx.gpu[:num_reqs_padded],
        group_key_cache_idx = self.group_key_cache_idx.gpu[:num_reqs_padded],
    )

    if logits_indices is not None and self.cache_config.kv_sharing_fast_prefill:
        cm_base.num_logits_indices = logits_indices.size(0)
        cm_base.logits_indices_padded = self._prepare_kv_sharing_fast_prefill(logits_indices)

    def _build_attn_group_metadata(
        kv_cache_gid: int,
        attn_gid: int,
        common_attn_metadata: CommonAttentionMetadata,
        prefill_ratio_to_sas_metadata: dict,
        decode_ratio_to_sas_metadata: dict,
        common_ratio_to_sas_metadata: dict,
        ubid: int | None = None,
    ) -> None:
        attn_group = self.attn_groups[kv_cache_gid][attn_gid]
        builder = attn_group.get_metadata_builder(ubid or 0)
        cascade_attn_prefix_len = (
            cascade_attn_prefix_lens[kv_cache_gid][attn_gid] if cascade_attn_prefix_lens else 0
        )

        extra_attn_metadata_args = {}
        if use_spec_decode and isinstance(builder, GDNAttentionMetadataBuilder):
            assert ubid is None, "UBatching not supported with GDN yet"
            extra_attn_metadata_args = dict(
                num_accepted_tokens=self.num_accepted_tokens.gpu[:num_reqs_padded],
                num_decode_draft_tokens_cpu=self.num_decode_draft_tokens.cpu[:num_reqs_padded],
            )

        if isinstance(builder, (AscendDSAMetadataBuilder, AscendDSACPMetadataBuilder)):
            if for_cudagraph_capture:
                prefill_ratio_to_sas_metadata = {}
                decode_ratio_to_sas_metadata = {}
                common_ratio_to_sas_metadata = {}
            extra_attn_metadata_args = dict(
                num_reqs_actual=num_reqs,
                prefill_ratio_to_sas_metadata=prefill_ratio_to_sas_metadata,
                decode_ratio_to_sas_metadata=decode_ratio_to_sas_metadata,
                common_ratio_to_sas_metadata=common_ratio_to_sas_metadata,
                block_size=attn_group.kv_cache_spec.block_size,
            )

        if (for_cudagraph_capture
                and not isinstance(builder, (
                    AscendDSAMetadataBuilder,
                    AscendDSACPMetadataBuilder,
                    AscendSFADCPMetadataBuilder,
                ))):
            attn_metadata_i = builder.build_for_cudagraph_capture(common_attn_metadata)
        else:
            attn_metadata_i = builder.build(
                common_prefix_len=cascade_attn_prefix_len,
                common_attn_metadata=common_attn_metadata,
                **extra_attn_metadata_args,
            )
            # NOTE(zxr): Due to the Triton operator does not deal with -1 padding in FullGraph mode,
            # the padding needs to be changed from -1 to 0 to avoid writing invalid mamba block.
            if self.vllm_config.compilation_config.cudagraph_mode.has_full_cudagraphs() \
                and isinstance(builder, GDNAttentionMetadataBuilder) and attn_metadata_i.num_prefills == 0:
                if attn_metadata_i.num_decodes == 0 and attn_metadata_i.num_spec_decodes > 0:
                    attn_metadata_i.spec_state_indices_tensor[attn_metadata_i.num_spec_decodes:].fill_(0)
        if isinstance(builder, AscendDSAMetadataBuilder):
            prefill_ratio_to_sas_metadata = builder.prefill_ratio_to_sas_metadata  # type: ignore[assignment]
            decode_ratio_to_sas_metadata = builder.decode_ratio_to_sas_metadata  # type: ignore[assignment]
            common_ratio_to_sas_metadata = builder.common_ratio_to_sas_metadata  # type: ignore[assignment]

        if ubid is None:
            assert isinstance(attn_metadata, dict)
            attn_metadata_dict = attn_metadata
        else:
            assert isinstance(attn_metadata, list)
            attn_metadata_dict = attn_metadata[ubid]

        for layer_name in attn_group.layer_names:
            attn_metadata_dict[layer_name] = attn_metadata_i

    # Prepare the attention metadata for each KV cache group and make layers
    # in the same group share the same metadata.
    prefill_ratio_to_sas_metadata: dict[Any, Any] = {}
    decode_ratio_to_sas_metadata: dict[Any, Any] = {}
    common_ratio_to_sas_metadata: dict[Any, Any] = {}

    # ascend vllm adaptor start
    # Defer CPU seq_lens correction to overlap the previous count D2H with
    # input preparation and metadata construction for eligible GDN groups.
    # Correct once before a consumer needs exact CPU lengths (earlier for DCP).
    # Reorder metadata construction only; preserve original group order for
    # drafter bookkeeping and selection.
    spec_decode_common_attn_metadata = None

    def update_drafter_metadata(kv_cache_gid, kv_cache_group, cm):
        nonlocal spec_decode_common_attn_metadata

        if self.speculative_config and isinstance(
            self.drafter,
            (AscendStep3p5MTPProposer, AscendDSparkProposer),
        ):
            self.drafter.set_per_group_attn_metadata(
                kv_cache_gid,
                cm.block_table_tensor,
                cm.slot_mapping,
            )

        if (
            self.speculative_config
            and spec_decode_common_attn_metadata is None
        ):
            if isinstance(
                self.drafter,
                (
                    AscendEagleProposer,
                    AscendDraftModelProposer,
                    AscendDflashProposer,
                    AscendDSparkProposer,
                ),
            ):
                if self.drafter.attn_layer_names[0] in kv_cache_group.layer_names:
                    spec_decode_common_attn_metadata = cm
            else:
                spec_decode_common_attn_metadata = cm

    zero_bubble_enabled = bool(envs_ascend.ENABLE_ZERO_BUBBLE)

    logical_gids = list(range(len(kv_cache_groups)))
    build_gids = logical_gids

    # Reorder construction only; preserve original IDs for drafter selection.
    if correction_pending:
        can_build_early = {
            gid: self._can_build_group_before_seq_lens_correction(gid)
            for gid in logical_gids
        }
        # Build eligible GDN groups first; stable sorting preserves relative order.
        # If no group can build early, the original order is unchanged.
        build_gids = sorted(
            logical_gids,
            key=lambda gid: not can_build_early[gid],
        )

    common_metadata_by_gid = {}

    for kv_cache_gid in build_gids:
        if correction_pending and not can_build_early[kv_cache_gid]:
            correct_seq_lens_cpu_if_pending()
            cm_base.max_seq_len = int(
                self.optimistic_seq_lens_cpu.numpy()[:num_reqs].max()
            )

        kv_cache_group = kv_cache_groups[kv_cache_gid]
        cm = copy(cm_base)

        cm.encoder_seq_lens, cm.encoder_seq_lens_cpu = (
            self._get_encoder_seq_lens(
                num_scheduled_tokens or {},
                kv_cache_group.kv_cache_spec,
                num_reqs_padded,
            )
        )

        if self._has_gdn:
            attn_group = self.attn_groups[kv_cache_gid][0]
            builder = attn_group.get_metadata_builder(0)
            if isinstance(builder, GDNAttentionMetadataBuilder):
                cm.query_start_loc_cpu = (
                    self.gdn_query_start_loc.cpu[: num_reqs_padded + 1]
                )
                cm.query_start_loc = (
                    self.gdn_query_start_loc.gpu[: num_reqs_padded + 1]
                )

        if kv_cache_gid > 0:
            cm.block_table_tensor, cm.slot_mapping = (
                _get_block_table_and_slot_mapping(kv_cache_gid)
            )

        if zero_bubble_enabled:
            common_metadata_by_gid[kv_cache_gid] = cm
        else:
            # Preserve the original ordering: register with the drafter before build.
            update_drafter_metadata(kv_cache_gid, kv_cache_group, cm)

        for attn_gid in range(len(self.attn_groups[kv_cache_gid])):
            _build_attn_group_metadata(
                kv_cache_gid,
                attn_gid,
                cm,
                prefill_ratio_to_sas_metadata,
                decode_ratio_to_sas_metadata,
                common_ratio_to_sas_metadata,
            )

    if zero_bubble_enabled:
        correct_seq_lens_cpu_if_pending()

        if correction_applied:
            corrected_max_seq_len = int(
                self.optimistic_seq_lens_cpu.numpy()[:num_reqs].max()
            )
            cm_base.max_seq_len = corrected_max_seq_len

            # Scalar updates are not shared by shallow copies.
            for cm in common_metadata_by_gid.values():
                cm.max_seq_len = corrected_max_seq_len

        for kv_cache_gid in logical_gids:
            update_drafter_metadata(
                kv_cache_gid,
                kv_cache_groups[kv_cache_gid],
                common_metadata_by_gid[kv_cache_gid],
            )
    # ascend vllm adaptor end
    
    if self.is_mm_prefix_lm:
        req_doc_ranges = {}
        for req_id in self.input_batch.req_ids:
            image_doc_ranges = []
            req_state = self.requests[req_id]
            for mm_feature in req_state.mm_features:
                pos_info = mm_feature.mm_position
                img_doc_range = pos_info.extract_embeds_range()
                image_doc_ranges.extend(img_doc_range)
            req_idx = self.input_batch.req_id_to_index[req_id]
            req_doc_ranges[req_idx] = image_doc_ranges

        if isinstance(attn_metadata, list):
            for ub_metadata in attn_metadata:
                for _metadata in ub_metadata.values():
                    _metadata.mm_prefix_range = req_doc_ranges  # type: ignore[attr-defined]
        else:
            for _metadata in attn_metadata.values():
                _metadata.mm_prefix_range = req_doc_ranges  # type: ignore[attr-defined]

    if spec_decode_common_attn_metadata is not None and (
        num_reqs != num_reqs_padded or num_tokens != num_tokens_padded
    ):
        # Currently the drafter still only uses piecewise cudagraphs (and modifies
        # the attention metadata in directly), and therefore does not want to use
        # padded attention metadata.
        spec_decode_common_attn_metadata = spec_decode_common_attn_metadata.unpadded(num_tokens, num_reqs)
    return attn_metadata, spec_decode_common_attn_metadata

def _can_build_group_before_seq_lens_correction(self, kv_cache_gid: int) -> bool:
    attn_groups = self.attn_groups[kv_cache_gid]

    # Only nonempty, all-GDN groups can build before CPU correction.
    return bool(attn_groups) and all(
        isinstance(
            attn_group.get_metadata_builder(0),
            GDNAttentionMetadataBuilder,
        )
        for attn_group in attn_groups
    )

def _patch_moe_dp_metadata_sync(NPUModelRunner) -> None:
    """Keep MoE communication and ACLGraph modes consistent across DP ranks."""
    original_sync_metadata_across_dp = NPUModelRunner._sync_metadata_across_dp

    if getattr(
        original_sync_metadata_across_dp,
        _PATCH_MARKER,
        False,
    ):
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
            and not is_draft_model
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

        # Preserve each rank's local token count. Downstream code uses the
        # global maximum only to choose one consistent MoE communication mode.
        return (
            max_tokens_across_dp,
            tokens_across_dp,
            synced_cudagraph_mode,
        )

    setattr(
        sync_metadata_across_dp_with_moe_consistency,
        _PATCH_MARKER,
        True,
    )

    NPUModelRunner._sync_metadata_across_dp = sync_metadata_across_dp_with_moe_consistency


def apply_patch() -> None:
    global _PATCH_APPLIED

    if _PATCH_APPLIED:
        return

    from vllm_ascend.worker.model_runner_v1 import NPUModelRunner

    _patch_moe_dp_metadata_sync(NPUModelRunner)

    # Feature flags control branches inside the patched methods.
    NPUModelRunner._can_build_group_before_seq_lens_correction = (
        _can_build_group_before_seq_lens_correction
    )
    NPUModelRunner._prepare_inputs = _prepare_inputs
    NPUModelRunner._build_attention_metadata = _build_attention_metadata

    _PATCH_APPLIED = True


apply_patch()