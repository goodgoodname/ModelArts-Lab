from __future__ import annotations

import importlib.util
import sys
import types
import uuid
from pathlib import Path
from typing import Any

import pytest

ROOT = Path(__file__).resolve().parents[3]
PATCH_PATH = ROOT / "ascend_vllm" / "patch" / "worker" / "patch_mooncake_hybrid_connector.py"


def _make_package(name: str) -> types.ModuleType:
    module = types.ModuleType(name)
    module.__path__ = []  # type: ignore[attr-defined]
    return module


def _install_vllm_stubs(monkeypatch: pytest.MonkeyPatch) -> types.ModuleType:
    vllm = _make_package("vllm")
    logger_module = types.ModuleType("vllm.logger")

    class _Logger:
        def exception(self, *args: Any, **kwargs: Any) -> None:
            pass

    def init_logger(name: str) -> _Logger:
        return _Logger()

    logger_module.init_logger = init_logger  # type: ignore[attr-defined]

    vllm_ascend = _make_package("vllm_ascend")
    distributed = _make_package("vllm_ascend.distributed")
    kv_transfer = _make_package("vllm_ascend.distributed.kv_transfer")
    kv_p2p = _make_package("vllm_ascend.distributed.kv_transfer.kv_p2p")
    mooncake_hybrid_connector = types.ModuleType("vllm_ascend.distributed.kv_transfer.kv_p2p.mooncake_hybrid_connector")

    class KVCacheRecvingThread:
        def __init__(self) -> None:
            self.origin_init_called = True

        def _transfer_kv_cache(self, req_meta: dict[str, Any]) -> None:
            raise RuntimeError("single-group transfer failed")

        def _transfer_kv_cache_all_groups(self, req_meta: dict[str, Any]) -> None:
            raise RuntimeError("multi-group transfer failed")

    class MooncakeConnector:
        pass

    class MooncakeConnectorWorker:
        pass

    mooncake_hybrid_connector.KVCacheRecvingThread = KVCacheRecvingThread  # type: ignore[attr-defined]
    mooncake_hybrid_connector.MooncakeConnector = MooncakeConnector  # type: ignore[attr-defined]
    mooncake_hybrid_connector.MooncakeConnectorWorker = MooncakeConnectorWorker  # type: ignore[attr-defined]
    kv_p2p.mooncake_hybrid_connector = mooncake_hybrid_connector  # type: ignore[attr-defined]

    monkeypatch.setitem(sys.modules, "vllm", vllm)
    monkeypatch.setitem(sys.modules, "vllm.logger", logger_module)
    monkeypatch.setitem(sys.modules, "vllm_ascend", vllm_ascend)
    monkeypatch.setitem(sys.modules, "vllm_ascend.distributed", distributed)
    monkeypatch.setitem(sys.modules, "vllm_ascend.distributed.kv_transfer", kv_transfer)
    monkeypatch.setitem(sys.modules, "vllm_ascend.distributed.kv_transfer.kv_p2p", kv_p2p)
    monkeypatch.setitem(
        sys.modules,
        "vllm_ascend.distributed.kv_transfer.kv_p2p.mooncake_hybrid_connector",
        mooncake_hybrid_connector,
    )

    return mooncake_hybrid_connector


def _load_patch_module(monkeypatch: pytest.MonkeyPatch) -> tuple[Any, types.ModuleType]:
    mooncake_hybrid_connector = _install_vllm_stubs(monkeypatch)

    module_name = f"patch_mooncake_hybrid_connector_under_test_{uuid.uuid4().hex}"
    spec = importlib.util.spec_from_file_location(module_name, PATCH_PATH)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    monkeypatch.setitem(sys.modules, module_name, module)
    spec.loader.exec_module(module)
    return module, mooncake_hybrid_connector


def test_iter_block_ids_supports_flat_and_grouped_shapes(monkeypatch: pytest.MonkeyPatch) -> None:
    module, _ = _load_patch_module(monkeypatch)

    assert list(module._iter_block_ids([1, 2, 3])) == [1, 2, 3]
    assert list(module._iter_block_ids(([10, 11], None, [20, None]))) == [10, 11, 20]
    assert list(module._iter_block_ids(())) == []


def test_recv_thread_init_adds_failure_tracking_state(monkeypatch: pytest.MonkeyPatch) -> None:
    _, mhc = _load_patch_module(monkeypatch)

    recv_thread = mhc.KVCacheRecvingThread()

    assert recv_thread.origin_init_called is True
    assert recv_thread.invalid_block_ids == set()
    assert hasattr(recv_thread.failed_recv_requests_lock, "acquire")


def test_transfer_wrapper_marks_failed_local_blocks_and_reraises(monkeypatch: pytest.MonkeyPatch) -> None:
    _, mhc = _load_patch_module(monkeypatch)
    recv_thread = mhc.KVCacheRecvingThread()

    with pytest.raises(RuntimeError, match="multi-group transfer failed"):
        recv_thread._transfer_kv_cache_all_groups({"local_block_ids": ([1, 2], [101, None])})

    assert recv_thread.get_and_clear_invalid_block_ids() == {1, 2, 101}
    assert recv_thread.get_and_clear_invalid_block_ids() == set()


def test_transfer_wrapper_initializes_failure_state_for_existing_instances(monkeypatch: pytest.MonkeyPatch) -> None:
    _, mhc = _load_patch_module(monkeypatch)
    recv_thread = mhc.KVCacheRecvingThread.__new__(mhc.KVCacheRecvingThread)

    with pytest.raises(RuntimeError, match="single-group transfer failed"):
        recv_thread._transfer_kv_cache({"local_block_ids": [7, 8]})

    assert recv_thread.get_and_clear_invalid_block_ids() == {7, 8}


def test_connector_and_worker_forward_load_errors_from_decode_recv_thread(monkeypatch: pytest.MonkeyPatch) -> None:
    _, mhc = _load_patch_module(monkeypatch)

    connector = mhc.MooncakeConnector()
    connector.connector_worker = types.SimpleNamespace(get_block_ids_with_load_errors=lambda: {3, 5})
    assert connector.get_block_ids_with_load_errors() == {3, 5}

    worker = mhc.MooncakeConnectorWorker()
    worker.kv_role = "kv_consumer"
    worker.kv_recv_thread = types.SimpleNamespace(get_and_clear_invalid_block_ids=lambda: {8, 13})
    assert worker.get_block_ids_with_load_errors() == {8, 13}

    worker.kv_role = "kv_producer"
    assert worker.get_block_ids_with_load_errors() == set()
