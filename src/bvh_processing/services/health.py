"""Cached, bounded GPU checks; HTTP liveness never waits for CUDA."""
from __future__ import annotations

import asyncio
from contextlib import suppress
from datetime import datetime, timezone
import json
import logging
import os
from pathlib import Path
import shutil
import sys
import time

logger = logging.getLogger(__name__)
GPU_PROBE_SCRIPT = '''
import json
import torch
if not torch.cuda.is_available():
    print(json.dumps({"status": "unavailable", "compute_ok": False}))
else:
    with torch.inference_mode():
        x = torch.ones((32, 32), device="cuda:0")
        y = x @ x
        torch.cuda.synchronize()
        ok = bool((y == 32).all().item())
        free, total = torch.cuda.mem_get_info(0)
        print(json.dumps({"status": "ok" if ok else "error", "compute_ok": ok,
            "device": torch.cuda.get_device_name(0), "cuda_version": torch.version.cuda,
            "torch_version": torch.__version__, "memory_free_bytes": free,
            "memory_total_bytes": total}))
'''


async def probe_gpu(timeout: float) -> dict:
    """Isolate CUDA initialization failures/hangs and always reap the process."""
    process = await asyncio.create_subprocess_exec(
        sys.executable, "-c", GPU_PROBE_SCRIPT,
        stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.DEVNULL,
    )
    try:
        stdout, _ = await asyncio.wait_for(process.communicate(), timeout)
        if process.returncode != 0:
            return {"status": "error", "compute_ok": False, "error": "probe_process_failed"}
        result = json.loads(stdout)
        if not isinstance(result, dict) or not isinstance(result.get("compute_ok"), bool):
            raise ValueError("Invalid GPU probe response")
        return result
    except asyncio.TimeoutError:
        return {"status": "timeout", "compute_ok": False}
    except (ValueError, UnicodeError):
        return {"status": "error", "compute_ok": False, "error": "invalid_probe_response"}
    finally:
        if process.returncode is None:
            with suppress(ProcessLookupError):
                process.kill()
        await process.wait()


class HealthMonitor:
    def __init__(self, settings):
        self.settings = settings
        self.started = time.monotonic()
        self.last_finished: float | None = None
        self.result: dict = {"status": "checking" if settings.health_gpu_enabled else "disabled", "compute_ok": False}
        self.task: asyncio.Task | None = None
        self.check_count = 0
        self.consecutive_failures = 0
        self.last_success_at: str | None = None

    def start(self) -> None:
        if self.settings.health_gpu_enabled:
            self.task = asyncio.create_task(self._run(), name="gpu-health-monitor")

    async def close(self) -> None:
        if self.task is not None:
            self.task.cancel()
            with suppress(asyncio.CancelledError):
                await self.task

    async def _run(self) -> None:
        while True:
            start = time.monotonic()
            try:
                result = await probe_gpu(self.settings.health_gpu_timeout_seconds)
            except Exception:
                logger.exception("GPU health probe failed")
                result = {"status": "error", "compute_ok": False, "error": "probe_failed"}
            result["checked_at"] = datetime.now(timezone.utc).isoformat()
            result["duration_seconds"] = round(time.monotonic() - start, 3)
            self.check_count += 1
            if result.get("compute_ok") and result.get("status") == "ok":
                self.consecutive_failures = 0
                self.last_success_at = result["checked_at"]
            else:
                self.consecutive_failures += 1
            previous = self.result["status"]
            self.result = result
            self.last_finished = time.monotonic()
            if result["status"] != previous:
                logger.log(logging.INFO if result["status"] == "ok" else logging.WARNING,
                           "GPU health changed: %s -> %s", previous, result["status"])
            await asyncio.sleep(self.settings.health_gpu_interval_seconds)

    def snapshot(self) -> dict:
        age = None if self.last_finished is None else time.monotonic() - self.last_finished
        stale = age is not None and age > self.settings.health_gpu_interval_seconds + self.settings.health_gpu_timeout_seconds + 10
        gpu = {**self.result, "age_seconds": None if age is None else round(age, 3),
               "stale": stale, "check_count": self.check_count,
               "consecutive_failures": self.consecutive_failures,
               "last_success_at": self.last_success_at}
        module = sys.modules.get("bvh_processing.services.mdm_transition")
        engine = getattr(module, "_MDM_ENGINE", None)
        loaded = engine is not None and engine.model is not None
        model = {"loaded": loaded, "device": str(engine.device) if engine is not None else None}
        gpu_ok = gpu["status"] == "ok" and gpu["compute_ok"] and not stale
        model_ok = not loaded or str(engine.device).startswith("cuda")
        ready = not self.settings.health_gpu_required or (gpu_ok and model_ok)
        resources = {"cpu_count": os.cpu_count()}
        with suppress(OSError, ValueError):
            resources["host_load_average"] = list(os.getloadavg())
        with suppress(OSError, ValueError):
            for line in Path("/proc/self/status").read_text().splitlines():
                if line.startswith("VmRSS:"):
                    resources["process_rss_bytes"] = int(line.split()[1]) * 1024
        for key, filename in (("container_memory_current_bytes", "memory.current"),
                              ("container_memory_limit_bytes", "memory.max")):
            with suppress(OSError, ValueError):
                value = Path("/sys/fs/cgroup", filename).read_text().strip()
                resources[key] = None if value == "max" else int(value)
        with suppress(OSError):
            disk = shutil.disk_usage("/tmp")
            resources["tmp_filesystem"] = {"total_bytes": disk.total, "free_bytes": disk.free}
        return {"status": "ok" if ready else "degraded", "ready": ready,
                "uptime_seconds": round(time.monotonic() - self.started, 3),
                "gpu_required": self.settings.health_gpu_required, "gpu": gpu,
                "mdm": model, "resources": resources,
                "probe_interval_seconds": self.settings.health_gpu_interval_seconds,
                "probe_timeout_seconds": self.settings.health_gpu_timeout_seconds}
