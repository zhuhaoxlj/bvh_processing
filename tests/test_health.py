import asyncio
import sys
from types import SimpleNamespace

from fastapi.testclient import TestClient

from bvh_processing.config import Settings
from bvh_processing.main import create_app
from bvh_processing.services import health


def settings(**kwargs):
    return Settings(_env_file=None, **kwargs)


def test_liveness_compatible_and_readiness_initializing():
    with TestClient(create_app(settings(health_gpu_enabled=False))) as client:
        assert client.get('/health').json() == {'status': 'ok'}
        details = client.get('/health/details').json()
        assert details['gpu']['status'] == 'disabled'
        assert details['ready'] is False
        assert 'process_rss_bytes' in details['resources']
        assert client.get('/health/ready').status_code == 503
        monitor = client.app.state.health_monitor
        monitor.result = {'status': 'ok', 'compute_ok': True}
        monitor.last_finished = health.time.monotonic()
        assert client.get('/health/ready').status_code == 200
        monitor.result = {'status': 'timeout', 'compute_ok': False}
        assert client.get('/health/ready').status_code == 503


def test_cpu_only_readiness_can_be_enabled():
    with TestClient(create_app(settings(health_gpu_enabled=False, health_gpu_required=False))) as client:
        assert client.get('/health/ready').status_code == 200


def test_cached_failure_stale_and_cpu_model(monkeypatch):
    monitor = health.HealthMonitor(settings())
    monitor.result = {'status': 'ok', 'compute_ok': True}
    monitor.last_finished = health.time.monotonic()
    assert monitor.snapshot()['ready']
    monitor.last_finished -= 1000
    assert monitor.snapshot()['gpu']['stale']
    assert not monitor.snapshot()['ready']
    monitor.last_finished = health.time.monotonic()
    monkeypatch.setitem(sys.modules, 'bvh_processing.services.mdm_transition',
                        SimpleNamespace(_MDM_ENGINE=SimpleNamespace(model=object(), device='cpu')))
    assert not monitor.snapshot()['ready']


def test_periodic_probe_recovers_and_shuts_down(monkeypatch):
    async def scenario():
        calls = []
        recovered = asyncio.Event()
        async def fake_probe(timeout):
            calls.append(timeout)
            if len(calls) == 1:
                return {'status': 'unavailable', 'compute_ok': False}
            recovered.set()
            return {'status': 'ok', 'compute_ok': True}
        monkeypatch.setattr(health, 'probe_gpu', fake_probe)
        config = settings()
        config.health_gpu_interval_seconds = 0.01
        monitor = health.HealthMonitor(config)
        monitor.start()
        await asyncio.wait_for(recovered.wait(), 1)
        assert monitor.snapshot()['ready']
        assert monitor.snapshot()['gpu']['checked_at']
        await monitor.close()
        assert monitor.task.done()
    asyncio.run(scenario())


def test_probe_timeout_and_cancellation(monkeypatch):
    monkeypatch.setattr(health, 'GPU_PROBE_SCRIPT', 'import time; time.sleep(60)')
    async def scenario():
        processes = []
        original = asyncio.create_subprocess_exec
        async def track(*args, **kwargs):
            process = await original(*args, **kwargs)
            processes.append(process)
            return process
        monkeypatch.setattr(asyncio, 'create_subprocess_exec', track)
        assert (await health.probe_gpu(0.1))['status'] == 'timeout'
        task = asyncio.create_task(health.probe_gpu(60))
        await asyncio.sleep(0.1)
        task.cancel()
        try:
            await task
        except asyncio.CancelledError:
            pass
        assert task.cancelled()
        assert len(processes) == 2
        assert all(process.returncode is not None for process in processes)
    asyncio.run(scenario())


def test_probe_success_and_malformed_result(monkeypatch):
    async def scenario():
        monkeypatch.setattr(health, 'GPU_PROBE_SCRIPT', 'print(\'{"status":"ok","compute_ok":true}\')')
        assert (await health.probe_gpu(3))['compute_ok']
        monkeypatch.setattr(health, 'GPU_PROBE_SCRIPT', 'print("invalid")')
        assert (await health.probe_gpu(3))['status'] == 'error'
    asyncio.run(scenario())
