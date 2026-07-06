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


def _make_package(name: str) -> types.ModuleType:
    module = types.ModuleType(name)
    module.__path__ = []  # type: ignore[attr-defined]
    return module


class _Array:
    def __init__(self, values: list[int]) -> None:
        self.values = values

    def reshape(self, shape: tuple[int, int]) -> _Matrix:
        rows, cols = shape
        if rows == 1:
            return _Matrix([self.values[:cols]])

        return _Matrix([[value] for value in self.values[:rows]])

    def flatten(self) -> _Array:
        return self

    def tolist(self) -> list[int]:
        return self.values

    def __getitem__(self, item: slice) -> _Array:
        return _Array(self.values[item])


class _Matrix:
    def __init__(self, values: list[list[int]]) -> None:
        self.values = values

    def __mul__(self, scalar: int) -> _Matrix:
        return _Matrix([[value * scalar for value in row] for row in self.values])

    def __add__(self, other: _Matrix) -> _Matrix:
        if len(self.values) == 1:
            return _Matrix([[left + right[0] for left in self.values[0]] for right in other.values])

        return _Matrix([[left[0] + right for right in other.values[0]] for left in self.values])

    def flatten(self) -> _Array:
        return _Array([value for row in self.values for value in row])


def _install_numpy_stub(monkeypatch: pytest.MonkeyPatch) -> None:
    numpy = types.ModuleType("numpy")
    numpy.int32 = int  # type: ignore[attr-defined]
    numpy.array = lambda values, dtype=None: _Array(list(values))  # type: ignore[attr-defined]
    numpy.arange = lambda start, stop: _Array(list(range(start, stop)))  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "numpy", numpy)


def _install_vllm_stubs(monkeypatch: pytest.MonkeyPatch) -> types.SimpleNamespace:
    _install_numpy_stub(monkeypatch)

    vllm = _make_package("vllm")
    v1 = _make_package("vllm.v1")
    core = _make_package("vllm.v1.core")
    sched = _make_package("vllm.v1.core.sched")
    scheduler_module = types.ModuleType("vllm.v1.core.sched.scheduler")

    class Scheduler:
        def update_from_output(self, scheduler_output: Any, model_runner_output: Any) -> set[str]:
            return self._handle_invalid_blocks({99})

        def _handle_invalid_blocks(
            self,
            invalid_block_ids: set[int],
            num_scheduled_tokens: dict[str, int],
        ) -> set[str]:
            self.handle_invalid_args = (invalid_block_ids, num_scheduled_tokens)
            return {"handled"}

    scheduler_module.Scheduler = Scheduler  # type: ignore[attr-defined]

    vllm_ascend = _make_package("vllm_ascend")
    ascend_core = _make_package("vllm_ascend.core")
    recompute_scheduler = types.ModuleType("vllm_ascend.core.recompute_scheduler")

    class RecomputeScheduler(Scheduler):
        pass

    class AsyncRecomputeScheduler(Scheduler):
        pass

    recompute_scheduler.RecomputeScheduler = RecomputeScheduler  # type: ignore[attr-defined]
    recompute_scheduler.AsyncRecomputeScheduler = AsyncRecomputeScheduler  # type: ignore[attr-defined]
    ascend_core.recompute_scheduler = recompute_scheduler  # type: ignore[attr-defined]

    monkeypatch.setitem(sys.modules, "vllm", vllm)
    monkeypatch.setitem(sys.modules, "vllm.v1", v1)
    monkeypatch.setitem(sys.modules, "vllm.v1.core", core)
    monkeypatch.setitem(sys.modules, "vllm.v1.core.sched", sched)
    monkeypatch.setitem(sys.modules, "vllm.v1.core.sched.scheduler", scheduler_module)
    monkeypatch.setitem(sys.modules, "vllm_ascend", vllm_ascend)
    monkeypatch.setitem(sys.modules, "vllm_ascend.core", ascend_core)
    monkeypatch.setitem(sys.modules, "vllm_ascend.core.recompute_scheduler", recompute_scheduler)

    return types.SimpleNamespace(
        Scheduler=Scheduler,
        RecomputeScheduler=RecomputeScheduler,
        AsyncRecomputeScheduler=AsyncRecomputeScheduler,
    )


def _load_patch_module(monkeypatch: pytest.MonkeyPatch) -> tuple[Any, types.SimpleNamespace]:
    stubs = _install_vllm_stubs(monkeypatch)

    module_name = f"patch_recompute_scheduler_under_test_{uuid.uuid4().hex}"
    spec = importlib.util.spec_from_file_location(module_name, PATCH_PATH)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    monkeypatch.setitem(sys.modules, module_name, module)
    spec.loader.exec_module(module)
    return module, stubs


def test_update_requests_with_invalid_blocks_handles_multi_group_blocks(monkeypatch: pytest.MonkeyPatch) -> None:
    _, stubs = _load_patch_module(monkeypatch)
    scheduler: Any = stubs.RecomputeScheduler()
    scheduler.block_size = 128
    scheduler.kv_cache_manager = types.SimpleNamespace(
        get_block_ids=lambda req_id: (
            [10, 11, 12],
            [20, 21, 22],
        )
    )
    request = types.SimpleNamespace(request_id="req1", num_computed_tokens=384)

    affected_req_ids, affected_tokens, blocks_to_evict = scheduler._update_requests_with_invalid_blocks(
        [request],
        invalid_block_ids={21},
        num_scheduled_tokens={"req1": 128},
    )

    assert affected_req_ids == {"req1"}
    assert affected_tokens == 128
    assert request.num_computed_tokens == 128
    assert blocks_to_evict == {11, 12, 21, 22}


def test_update_requests_with_invalid_blocks_ignores_scheduled_tail_blocks(monkeypatch: pytest.MonkeyPatch) -> None:
    _, stubs = _load_patch_module(monkeypatch)
    scheduler: Any = stubs.AsyncRecomputeScheduler()
    scheduler.block_size = 128
    scheduler.kv_cache_manager = types.SimpleNamespace(
        get_block_ids=lambda req_id: (
            [10, 11, 12],
            [20, 21, 22],
        )
    )
    request = types.SimpleNamespace(request_id="req1", num_computed_tokens=384)

    affected_req_ids, affected_tokens, blocks_to_evict = scheduler._update_requests_with_invalid_blocks(
        [request],
        invalid_block_ids={22},
        num_scheduled_tokens={"req1": 128},
    )

    assert affected_req_ids == set()
    assert affected_tokens == 0
    assert request.num_computed_tokens == 384
    assert blocks_to_evict == set()


def test_handle_invalid_blocks_uses_num_scheduled_tokens_from_scheduler_output(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _, stubs = _load_patch_module(monkeypatch)
    scheduler: Any = stubs.RecomputeScheduler()
    scheduler_output = types.SimpleNamespace(num_scheduled_tokens={"req1": 3})

    assert scheduler.update_from_output(scheduler_output, object()) == {"handled"}
    assert scheduler.handle_invalid_args == ({99}, {"req1": 3})
    assert not hasattr(scheduler, "_modelarts_current_num_scheduled_tokens")


def test_handle_invalid_blocks_requires_num_scheduled_tokens_without_scheduler_output(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _, stubs = _load_patch_module(monkeypatch)
    scheduler: Any = stubs.RecomputeScheduler()

    with pytest.raises(RuntimeError, match="num_scheduled_tokens is required"):
        scheduler._handle_invalid_blocks({99})

    assert scheduler._handle_invalid_blocks({99}, {"req1": 1}) == {"handled"}
    assert scheduler.handle_invalid_args == ({99}, {"req1": 1})


def test_get_routed_experts_uses_selected_kv_cache_group(monkeypatch: pytest.MonkeyPatch) -> None:
    _, stubs = _load_patch_module(monkeypatch)
    scheduler: Any = stubs.RecomputeScheduler()
    scheduler.vllm_config = types.SimpleNamespace(model_config=types.SimpleNamespace(enable_return_routed_experts=True))
    scheduler.routed_experts_attn_gid = 1
    scheduler.kv_cache_config = types.SimpleNamespace(
        kv_cache_groups=[
            types.SimpleNamespace(kv_cache_spec=types.SimpleNamespace(block_size=128)),
            types.SimpleNamespace(kv_cache_spec=types.SimpleNamespace(block_size=4)),
        ]
    )
    scheduler.kv_cache_manager = types.SimpleNamespace(
        get_blocks=lambda req_id: types.SimpleNamespace(get_block_ids=lambda: ([1], [4, 5]))
    )

    class RoutedExpertsReader:
        def get_routed_experts(self, indices: Any) -> list[int]:
            self.indices = indices
            return indices.tolist()

    scheduler.routed_experts_reader = RoutedExpertsReader()
    request = types.SimpleNamespace(request_id="req1", num_tokens=6)

    assert scheduler._get_routed_experts(request) == [16, 17, 18, 19, 20]
