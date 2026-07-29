from __future__ import annotations

_PATCH_APPLIED = False


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

            max_blocks = min(req_num_computed_blocks, max((len(group) for group in req_block_id_groups), default=0))

            for idx in range(max_blocks):
                block_ids_at_idx = [group[idx] for group in req_block_id_groups if idx < len(group)]

                invalid_block_ids_at_idx = [block_id for block_id in block_ids_at_idx if block_id in invalid_block_ids]
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

    rs.RecomputeScheduler._update_requests_with_invalid_blocks = update_requests_with_invalid_blocks

    if hasattr(rs, "AsyncRecomputeScheduler"):
        rs.AsyncRecomputeScheduler._update_requests_with_invalid_blocks = update_requests_with_invalid_blocks


def apply_patch() -> None:
    global _PATCH_APPLIED

    if _PATCH_APPLIED:
        return

    _patch_recompute_scheduler()
    _PATCH_APPLIED = True


apply_patch()
