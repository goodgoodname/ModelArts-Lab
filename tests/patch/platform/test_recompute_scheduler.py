from __future__ import annotations

import importlib.util
import sys
import types
import uuid
from pathlib import Path
from typing import Any

import pytest

ROOT = Path(__file__).resolve().parents[3]
PATCH_PATH = ROOT / "ascend_vllm" / "patch" / "platform" / "patch_recompute_scheduler.py"


class FakeRequest:
    def __init__(self, request_id: str, num_computed_tokens: int) -> None:
        self.request_id = request_id
        self.num_computed_tokens = num_computed_tokens


class FakeKVCacheManager:
    def __init__(self, block_ids_by_req: dict[str, tuple[list[int], ...]]) -> None:
        self.block_ids_by_req = block_ids_by_req

    def get_block_ids(self, req_id: str) -> tuple[list[int], ...]:
        return self.block_ids_by_req[req_id]


def _make_package(name: str) -> types.ModuleType:
    module = types.ModuleType(name)
    module.__path__ = []  # type: ignore[attr-defined]
    return module


def _install_recompute_scheduler_stub(monkeypatch: pytest.MonkeyPatch) -> types.ModuleType:
    vllm_ascend = _make_package("vllm_ascend")
    core = _make_package("vllm_ascend.core")
    recompute_scheduler = types.ModuleType("vllm_ascend.core.recompute_scheduler")

    class RecomputeScheduler:
        pass

    class AsyncRecomputeScheduler:
        pass

    vars(recompute_scheduler)["RecomputeScheduler"] = RecomputeScheduler
    vars(recompute_scheduler)["AsyncRecomputeScheduler"] = AsyncRecomputeScheduler
    vars(core)["recompute_scheduler"] = recompute_scheduler
    vars(vllm_ascend)["core"] = core

    monkeypatch.setitem(sys.modules, "vllm_ascend", vllm_ascend)
    monkeypatch.setitem(sys.modules, "vllm_ascend.core", core)
    monkeypatch.setitem(sys.modules, "vllm_ascend.core.recompute_scheduler", recompute_scheduler)

    return recompute_scheduler


def _load_patch_module(monkeypatch: pytest.MonkeyPatch) -> tuple[Any, types.ModuleType]:
    recompute_scheduler = _install_recompute_scheduler_stub(monkeypatch)

    module_name = f"patch_recompute_scheduler_under_test_{uuid.uuid4().hex}"
    spec = importlib.util.spec_from_file_location(module_name, PATCH_PATH)
    assert spec is not None
    assert spec.loader is not None

    module = importlib.util.module_from_spec(spec)
    monkeypatch.setitem(sys.modules, module_name, module)
    spec.loader.exec_module(module)

    return module, recompute_scheduler


def _make_scheduler(
    recompute_scheduler: types.ModuleType,
    block_ids_by_req: dict[str, tuple[list[int], ...]],
    block_size: int = 10,
) -> Any:
    scheduler_cls = recompute_scheduler.__dict__["RecomputeScheduler"]
    scheduler = scheduler_cls()
    scheduler.block_size = block_size
    scheduler.kv_cache_manager = FakeKVCacheManager(block_ids_by_req)
    return scheduler


def test_patch_installs_hma_invalid_block_handler_on_recompute_schedulers(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _, recompute_scheduler = _load_patch_module(monkeypatch)

    recompute_cls = recompute_scheduler.__dict__["RecomputeScheduler"]
    async_recompute_cls = recompute_scheduler.__dict__["AsyncRecomputeScheduler"]

    assert hasattr(recompute_cls, "_update_requests_with_invalid_blocks")
    assert (
        recompute_cls._update_requests_with_invalid_blocks
        is async_recompute_cls._update_requests_with_invalid_blocks
    )


def test_invalid_block_in_any_group_truncates_request_and_evicts_all_groups(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _, recompute_scheduler = _load_patch_module(monkeypatch)
    scheduler = _make_scheduler(
        recompute_scheduler,
        {
            "req-1": ([1, 2, 3, 4], [101, 102, 103]),
        },
    )
    request = FakeRequest("req-1", num_computed_tokens=40)

    affected_req_ids, total_affected_tokens, blocks_to_evict = scheduler._update_requests_with_invalid_blocks(
        [request],
        {102},
        {"req-1": 10},
    )

    assert affected_req_ids == {"req-1"}
    assert total_affected_tokens == 20
    assert blocks_to_evict == {2, 3, 4, 102, 103}
    assert request.num_computed_tokens == 10


def test_shared_invalid_blocks_are_only_rolled_back_once(monkeypatch: pytest.MonkeyPatch) -> None:
    _, recompute_scheduler = _load_patch_module(monkeypatch)
    scheduler = _make_scheduler(
        recompute_scheduler,
        {
            "req-a": ([1, 2, 3], [101, 102, 103]),
            "req-b": ([7, 2, 8], [107, 102, 108]),
        },
    )
    first_request = FakeRequest("req-a", num_computed_tokens=30)
    second_request = FakeRequest("req-b", num_computed_tokens=30)

    affected_req_ids, total_affected_tokens, blocks_to_evict = scheduler._update_requests_with_invalid_blocks(
        [first_request, second_request],
        {2, 102},
        {},
    )

    assert affected_req_ids == {"req-a", "req-b"}
    assert total_affected_tokens == 20
    assert blocks_to_evict == {2, 3, 102, 103}
    assert first_request.num_computed_tokens == 10
    assert second_request.num_computed_tokens == 30


def test_invalid_block_does_not_evict_when_evict_blocks_is_false(monkeypatch: pytest.MonkeyPatch) -> None:
    _, recompute_scheduler = _load_patch_module(monkeypatch)
    scheduler = _make_scheduler(
        recompute_scheduler,
        {
            "req-1": ([1, 2, 3, 4], [101, 102, 103]),
        },
    )
    request = FakeRequest("req-1", num_computed_tokens=40)

    affected_req_ids, total_affected_tokens, blocks_to_evict = scheduler._update_requests_with_invalid_blocks(
        [request],
        {102},
        {"req-1": 10},
        evict_blocks=False,
    )

    assert affected_req_ids == {"req-1"}
    assert total_affected_tokens == 20
    assert blocks_to_evict == set()
    assert request.num_computed_tokens == 10


def test_invalid_block_in_current_schedule_is_ignored(monkeypatch: pytest.MonkeyPatch) -> None:
    _, recompute_scheduler = _load_patch_module(monkeypatch)
    scheduler = _make_scheduler(
        recompute_scheduler,
        {
            "req-1": ([1, 2, 3, 4], [101, 102, 103, 104]),
        },
    )
    request = FakeRequest("req-1", num_computed_tokens=40)

    affected_req_ids, total_affected_tokens, blocks_to_evict = scheduler._update_requests_with_invalid_blocks(
        [request],
        {4, 104},
        {"req-1": 10},
    )

    assert affected_req_ids == set()
    assert total_affected_tokens == 0
    assert blocks_to_evict == set()
    assert request.num_computed_tokens == 40

def test_requests_without_invalid_blocks_are_unchanged(monkeypatch: pytest.MonkeyPatch) -> None:
    _, recompute_scheduler = _load_patch_module(monkeypatch)
    scheduler = _make_scheduler(
        recompute_scheduler,
        {
            "req-1": ([1, 2, 3], [101, 102, 103]),
        },
    )
    request = FakeRequest("req-1", num_computed_tokens=30)

    affected_req_ids, total_affected_tokens, blocks_to_evict = scheduler._update_requests_with_invalid_blocks(
        [request],
        {999},
        {},
    )

    assert affected_req_ids == set()
    assert total_affected_tokens == 0
    assert blocks_to_evict == set()
    assert request.num_computed_tokens == 30
