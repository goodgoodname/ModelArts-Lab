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


def _install_external_dependency_stubs(monkeypatch: pytest.MonkeyPatch) -> types.ModuleType:
    vllm = _make_package("vllm")
    vllm_logger = types.ModuleType("vllm.logger")

    class FakeLogger:
        def exception(self, *args: Any, **kwargs: Any) -> None:
            pass

    def init_logger(name: str) -> FakeLogger:
        return FakeLogger()

    vars(vllm_logger)["init_logger"] = init_logger
    vars(vllm)["logger"] = vllm_logger

    monkeypatch.setitem(sys.modules, "vllm", vllm)
    monkeypatch.setitem(sys.modules, "vllm.logger", vllm_logger)

    vllm_ascend = _make_package("vllm_ascend")
    distributed = _make_package("vllm_ascend.distributed")
    kv_transfer = _make_package("vllm_ascend.distributed.kv_transfer")
    kv_p2p = _make_package("vllm_ascend.distributed.kv_transfer.kv_p2p")
    mooncake_hybrid_connector = types.ModuleType(
        "vllm_ascend.distributed.kv_transfer.kv_p2p.mooncake_hybrid_connector"
    )

    class KVCacheRecvingThread:
        def __init__(self, marker: str = "original") -> None:
            self.marker = marker

        def _transfer_kv_cache(self, req_meta: dict[str, Any]) -> None:
            raise RuntimeError("simulated single transfer failure")

        def _transfer_kv_cache_all_groups(self, req_meta: dict[str, Any]) -> None:
            raise RuntimeError("simulated hybrid transfer failure")

    class MooncakeConnector:
        def __init__(self, connector_worker: Any | None = None) -> None:
            self.connector_worker = connector_worker

    class MooncakeConnectorWorker:
        def __init__(
            self,
            kv_role: str = "kv_consumer",
            kv_recv_thread: Any | None = None,
        ) -> None:
            self.kv_role = kv_role
            self.kv_recv_thread = kv_recv_thread

    vars(mooncake_hybrid_connector)["KVCacheRecvingThread"] = KVCacheRecvingThread
    vars(mooncake_hybrid_connector)["MooncakeConnector"] = MooncakeConnector
    vars(mooncake_hybrid_connector)["MooncakeConnectorWorker"] = MooncakeConnectorWorker

    vars(kv_p2p)["mooncake_hybrid_connector"] = mooncake_hybrid_connector
    vars(kv_transfer)["kv_p2p"] = kv_p2p
    vars(distributed)["kv_transfer"] = kv_transfer
    vars(vllm_ascend)["distributed"] = distributed

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
    mooncake_hybrid_connector = _install_external_dependency_stubs(monkeypatch)

    module_name = f"patch_mooncake_hybrid_connector_under_test_{uuid.uuid4().hex}"
    spec = importlib.util.spec_from_file_location(module_name, PATCH_PATH)
    assert spec is not None
    assert spec.loader is not None

    module = importlib.util.module_from_spec(spec)
    monkeypatch.setitem(sys.modules, module_name, module)
    spec.loader.exec_module(module)

    return module, mooncake_hybrid_connector


def test_iter_block_ids_flattens_supported_block_id_shapes(monkeypatch: pytest.MonkeyPatch) -> None:
    module, _ = _load_patch_module(monkeypatch)

    assert list(module._iter_block_ids(None)) == []
    assert list(module._iter_block_ids([1, 2, None])) == [1, 2]
    assert list(module._iter_block_ids(([3, 4], [101, None, 102]))) == [3, 4, 101, 102]


def test_hybrid_transfer_failure_records_grouped_local_block_ids(monkeypatch: pytest.MonkeyPatch) -> None:
    _, mooncake_hybrid_connector = _load_patch_module(monkeypatch)

    recv_cls = mooncake_hybrid_connector.__dict__["KVCacheRecvingThread"]
    recv_thread = recv_cls()

    with pytest.raises(RuntimeError, match="simulated hybrid transfer failure"):
        recv_thread._transfer_kv_cache_all_groups(
            {
                "local_block_ids": ([1, 2], [101, None, 102]),
            }
        )

    assert recv_thread.get_and_clear_invalid_block_ids() == {1, 2, 101, 102}
    assert recv_thread.get_and_clear_invalid_block_ids() == set()


def test_single_transfer_failure_records_flat_local_block_ids(monkeypatch: pytest.MonkeyPatch) -> None:
    _, mooncake_hybrid_connector = _load_patch_module(monkeypatch)

    recv_cls = mooncake_hybrid_connector.__dict__["KVCacheRecvingThread"]
    recv_thread = recv_cls()

    with pytest.raises(RuntimeError, match="simulated single transfer failure"):
        recv_thread._transfer_kv_cache(
            {
                "local_block_ids": [7, 8],
            }
        )

    assert recv_thread.get_and_clear_invalid_block_ids() == {7, 8}
    assert recv_thread.get_and_clear_invalid_block_ids() == set()


def test_connector_facade_reports_decode_worker_errors_once(monkeypatch: pytest.MonkeyPatch) -> None:
    _, mooncake_hybrid_connector = _load_patch_module(monkeypatch)

    recv_cls = mooncake_hybrid_connector.__dict__["KVCacheRecvingThread"]
    worker_cls = mooncake_hybrid_connector.__dict__["MooncakeConnectorWorker"]
    connector_cls = mooncake_hybrid_connector.__dict__["MooncakeConnector"]

    recv_thread = recv_cls()
    recv_thread._mark_failed_recv_request(([5], [105, 106]))

    worker = worker_cls(kv_role="kv_consumer", kv_recv_thread=recv_thread)
    connector = connector_cls(connector_worker=worker)

    assert connector.get_block_ids_with_load_errors() == {5, 105, 106}
    assert connector.get_block_ids_with_load_errors() == set()


def test_producer_worker_does_not_report_decode_load_errors(monkeypatch: pytest.MonkeyPatch) -> None:
    _, mooncake_hybrid_connector = _load_patch_module(monkeypatch)

    recv_cls = mooncake_hybrid_connector.__dict__["KVCacheRecvingThread"]
    worker_cls = mooncake_hybrid_connector.__dict__["MooncakeConnectorWorker"]

    recv_thread = recv_cls()
    recv_thread._mark_failed_recv_request(([9],))

    worker = worker_cls(kv_role="kv_producer", kv_recv_thread=recv_thread)

    assert worker.get_block_ids_with_load_errors() == set()
