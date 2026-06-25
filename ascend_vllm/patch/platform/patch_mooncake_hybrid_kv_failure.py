from __future__ import annotations

import functools
<<<<<<< Updated upstream:ascend_vllm/patch/platform/patch_mooncake_hybrid_kv_failure.py
import threading
from numbers import Integral
from typing import Any

import numpy as np
from vllm.logger import init_logger

logger = init_logger("vllm.ascend_vllm.patch.platform.mooncake_hybrid_kv_failure")
=======

import numpy as np
>>>>>>> Stashed changes:ascend_vllm/patch/platform/patch_recompute_scheduler.py

_PATCH_APPLIED = False


def _iter_block_ids(block_ids: Any):
    """Flatten BlockIds into individual block ids."""
    if not block_ids:
        return

    for group in block_ids:
        if group is None:
            continue

        if isinstance(group, Integral):
            yield int(group)
            continue

        for block_id in group:
            if block_id is not None:
                yield int(block_id)


def _patch_mooncake_hybrid_connector() -> None:
    """Patch MooncakeHybridConnector to report KV load failures."""
    from vllm_ascend.distributed.kv_transfer.kv_p2p import mooncake_hybrid_connector as mhc

    recv_cls = mhc.KVCacheRecvingThread

    if not getattr(recv_cls, "_modelarts_kv_failure_patch_applied", False):
        origin_init = recv_cls.__init__

        @functools.wraps(origin_init)
        def patched_init(self, *args, **kwargs):
            origin_init(self, *args, **kwargs)
            self.invalid_block_ids = set()
            self.failed_recv_requests_lock = threading.Lock()

        def ensure_failure_state(self) -> None:
            if not hasattr(self, "invalid_block_ids"):
                self.invalid_block_ids = set()
            if not hasattr(self, "failed_recv_requests_lock"):
                self.failed_recv_requests_lock = threading.Lock()

        def mark_failed_recv_request(self, local_block_ids) -> None:
            ensure_failure_state(self)
            with self.failed_recv_requests_lock:
                self.invalid_block_ids.update(_iter_block_ids(local_block_ids))

        def get_and_clear_invalid_block_ids(self) -> set[int]:
            ensure_failure_state(self)
            with self.failed_recv_requests_lock:
                invalid_block_ids = set(self.invalid_block_ids)
                self.invalid_block_ids.clear()
            return invalid_block_ids

        def wrap_transfer(method_name: str) -> None:
            origin_method = getattr(recv_cls, method_name, None)
            if origin_method is None:
                return
            if getattr(origin_method, "_modelarts_wrapped", False):
                return

            @functools.wraps(origin_method)
            def wrapped(self, req_meta, *args, **kwargs):
                try:
                    return origin_method(self, req_meta, *args, **kwargs)
                except Exception:
                    try:
                        self._mark_failed_recv_request(req_meta.get("local_block_ids", ()))
                    except Exception:
                        logger.exception("Failed to mark invalid KV blocks.")
                    raise

            wrapped._modelarts_wrapped = True
            setattr(recv_cls, method_name, wrapped)

        recv_cls.__init__ = patched_init
        recv_cls._mark_failed_recv_request = mark_failed_recv_request
        recv_cls.get_and_clear_invalid_block_ids = get_and_clear_invalid_block_ids

        wrap_transfer("_transfer_kv_cache")

        wrap_transfer("_transfer_kv_cache_all_groups")

        recv_cls._modelarts_kv_failure_patch_applied = True

    def connector_get_block_ids_with_load_errors(self) -> set[int]:
        assert self.connector_worker is not None
        return self.connector_worker.get_block_ids_with_load_errors()

    def worker_get_block_ids_with_load_errors(self) -> set[int]:
        if self.kv_role == "kv_consumer" and self.kv_recv_thread is not None:
            return self.kv_recv_thread.get_and_clear_invalid_block_ids()
        return set()

    mhc.MooncakeConnector.get_block_ids_with_load_errors = connector_get_block_ids_with_load_errors
    mhc.MooncakeConnectorWorker.get_block_ids_with_load_errors = worker_get_block_ids_with_load_errors


def _patch_recompute_scheduler() -> None:
    """Patch RecomputeScheduler for HMA invalid-block handling."""
    from vllm_ascend.core import recompute_scheduler as rs

    def update_requests_with_invalid_blocks(
        self,
        requests,
        invalid_block_ids: set[int],
        num_scheduled_tokens: dict[str, int],
        evict_blocks: bool = True,
    ) -> tuple[set[str], int, set[int]]:
        affected_req_ids: set[str] = set()
        total_affected_tokens = 0
        blocks_to_evict: set[int] = set()
        marked_invalid_block_ids: set[int] = set()

        for request in requests:
            is_affected = False
            marked_invalid_block = False
            req_id = request.request_id

            req_block_id_groups = self.kv_cache_manager.get_block_ids(req_id)
            req_num_computed_tokens = request.num_computed_tokens - num_scheduled_tokens.get(req_id, 0)
            req_num_computed_blocks = (req_num_computed_tokens + self.block_size - 1) // self.block_size

            max_blocks = min(
                req_num_computed_blocks,
                max((len(group) for group in req_block_id_groups), default=0),
            )

            for idx in range(max_blocks):
                block_ids_at_idx = [group[idx] for group in req_block_id_groups if idx < len(group)]

                invalid_block_ids_at_idx = [
                    block_id for block_id in block_ids_at_idx if block_id in invalid_block_ids
                ]
                if not invalid_block_ids_at_idx:
                    continue

                is_affected = True

                if all(block_id in marked_invalid_block_ids for block_id in invalid_block_ids_at_idx):
                    continue

                marked_invalid_block_ids.update(invalid_block_ids_at_idx)

                if marked_invalid_block:
                    continue

                marked_invalid_block = True

                request.num_computed_tokens = idx * self.block_size

                total_affected_tokens += req_num_computed_tokens - request.num_computed_tokens

                if evict_blocks:
                    for group in req_block_id_groups:
                        blocks_to_evict.update(group[idx:])

            if is_affected:
                if not marked_invalid_block:
                    total_affected_tokens += request.num_computed_tokens - req_num_computed_tokens
                    request.num_computed_tokens = req_num_computed_tokens

                affected_req_ids.add(request.request_id)

        return affected_req_ids, total_affected_tokens, blocks_to_evict

    def get_routed_experts(self, request):
        """Provide the upstream Scheduler routed-experts helper."""
        if not self.vllm_config.model_config.enable_return_routed_experts:
            return None

        kv_blocks = self.kv_cache_manager.get_blocks(request.request_id)
        block_ids = kv_blocks.get_block_ids()[self.routed_experts_attn_gid]
        num_tokens = request.num_tokens - 1

        block_ids_array = np.array(block_ids, dtype=np.int32)
        num_blocks = len(block_ids)

        attn_group = self.kv_cache_config.kv_cache_groups[self.routed_experts_attn_gid]
        block_size = attn_group.kv_cache_spec.block_size

        block_offsets = np.arange(0, block_size)
        slot_mapping = (
            block_offsets.reshape((1, block_size))
            + block_ids_array.reshape((num_blocks, 1)) * block_size
        ).flatten()[:num_tokens]

        return self.routed_experts_reader.get_routed_experts(indices=slot_mapping)

    def patch_invalid_blocks_signature(cls) -> None:
        """Make old RecomputeScheduler code compatible with vLLM 0.21.0 invalid-block handling."""
        token_attr = "_modelarts_current_num_scheduled_tokens"
        missing = object()

        origin_update_from_output = cls.update_from_output
        if not getattr(origin_update_from_output, "_modelarts_kv_failure_update_wrapped", False):

            @functools.wraps(origin_update_from_output)
            def patched_update_from_output(self, scheduler_output, model_runner_output):
                # Save num_scheduled_tokens for the duration of this scheduler update.
                previous_tokens = getattr(self, token_attr, missing)
                setattr(self, token_attr, scheduler_output.num_scheduled_tokens)

                try:
                    return origin_update_from_output(self, scheduler_output, model_runner_output)
                finally:
                    # Restore the previous value to avoid leaking state across scheduler steps.
                    if previous_tokens is missing:
                        if hasattr(self, token_attr):
                            delattr(self, token_attr)
                    else:
                        setattr(self, token_attr, previous_tokens)

            patched_update_from_output._modelarts_kv_failure_update_wrapped = True
            cls.update_from_output = patched_update_from_output

        origin_handle_invalid_blocks = cls._handle_invalid_blocks
        if not getattr(origin_handle_invalid_blocks, "_modelarts_kv_failure_handle_wrapped", False):

            @functools.wraps(origin_handle_invalid_blocks)
            def patched_handle_invalid_blocks(
                self,
                invalid_block_ids: set[int],
                num_scheduled_tokens: dict[str, int] | None = None,
            ) -> set[str]:
                # vLLM 0.21.0 requires num_scheduled_tokens, but the Ascend
                # RecomputeScheduler code may still call this method with one argument.
                if num_scheduled_tokens is None:
                    num_scheduled_tokens = getattr(self, token_attr, missing)
                    if num_scheduled_tokens is missing:
                        raise RuntimeError(
                            "num_scheduled_tokens is required when handling invalid KV blocks."
                        )

                return origin_handle_invalid_blocks(
                    self,
                    invalid_block_ids,
                    num_scheduled_tokens,
                )

            patched_handle_invalid_blocks._modelarts_kv_failure_handle_wrapped = True
            cls._handle_invalid_blocks = patched_handle_invalid_blocks

    rs.RecomputeScheduler._update_requests_with_invalid_blocks = update_requests_with_invalid_blocks
    rs.RecomputeScheduler._get_routed_experts = get_routed_experts
    patch_invalid_blocks_signature(rs.RecomputeScheduler)

    if hasattr(rs, "AsyncRecomputeScheduler"):
        rs.AsyncRecomputeScheduler._update_requests_with_invalid_blocks = update_requests_with_invalid_blocks
        rs.AsyncRecomputeScheduler._get_routed_experts = get_routed_experts
        patch_invalid_blocks_signature(rs.AsyncRecomputeScheduler)


def apply_patch() -> None:
    global _PATCH_APPLIED

    if _PATCH_APPLIED:
        return

    _patch_mooncake_hybrid_connector()
    _patch_recompute_scheduler()

<<<<<<< Updated upstream:ascend_vllm/patch/platform/patch_mooncake_hybrid_kv_failure.py
    _PATCH_APPLIED = True
    logger.info("Applied ModelArts Mooncake Hybrid KV failure monkey patch.")

=======
>>>>>>> Stashed changes:ascend_vllm/patch/platform/patch_recompute_scheduler.py

apply_patch()
