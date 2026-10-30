"""Run on bdca via SSH stdin. Replay task 1180 without business callbacks.
Only observations and temporary in-memory results; URLs/tokens are never printed.
Usage: python3 - baseline|repaired
"""
import json
import re
import subprocess
import sys

label = sys.argv[1]
logs = subprocess.run(['docker', 'logs', '--since', '3h', 'brss1-platform'], capture_output=True, text=True, check=True)
segments = None
for line in (logs.stdout + logs.stderr).splitlines():
    if 'UPDATE bvh_merge_task SET dance_id' in line and re.search(r'WHERE id = 1180\b', line):
        match = re.search(r"durations_json = '(.*?)', process_status", line)
        if match:
            segments = json.loads(match.group(1))
assert segments is not None, 'Task 1180 input not found in logs'
payload = {'actionId': 'diagnostic-replay-1180', 'segments': segments, 'callbackUrl': 'http://127.0.0.1:1/disabled'}
script = r'''
import asyncio, hashlib, json, time
from collections import defaultdict
import httpx, torch
from bvh_processing.config import get_settings
from bvh_processing.schemas import MergeBvhRequest
from bvh_processing.services import tasks, mdm_transition
import bvh_processing.vendor.mdm.merge as merge_module

payload = MergeBvhRequest.model_validate(PAYLOAD)
settings = get_settings()
print('BENCH_START', LABEL, 'cuda=', torch.cuda.is_available(), 'segments=', len(payload.segments), flush=True)
stats = defaultdict(float)
counts = defaultdict(int)
source_hashes = []
output_info = {}

def timed_sync(owner, name, key):
    original = getattr(owner, name)
    def wrapped(*args, **kwargs):
        start = time.perf_counter()
        try:
            return original(*args, **kwargs)
        finally:
            stats[key] += time.perf_counter() - start
            counts[key] += 1
    setattr(owner, name, wrapped)

for name in ['trim_bvh', 'normalize_bvh_frame_rates', 'adjust_bvh_motion_durations', 'merge_bvh_files']:
    timed_sync(tasks, name, name)
for name in ['convert_bvh', 'export_transition_bvh']:
    timed_sync(merge_module, name, name)
timed_sync(merge_module.MDMEngine, 'generate', 'model_generate_including_load')
original_download = tasks.download_bvh
async def download(*args, **kwargs):
    start = time.perf_counter()
    result = await original_download(*args, **kwargs)
    stats['download'] += time.perf_counter() - start
    result.content.seek(0)
    digest = hashlib.sha256(result.content.read()).hexdigest()
    result.content.seek(0)
    source_hashes.append(digest)
    return result
tasks.download_bvh = download
async def capture_callback(*args, **kwargs):
    output_info['success'] = kwargs.get('success')
    if not kwargs.get('success'):
        output_info['error'] = kwargs.get('message')
    file = kwargs.get('file')
    if file is not None:
        file.seek(0)
        raw = file.read()
        output_info['bytes'] = len(raw)
        output_info['sha256'] = hashlib.sha256(raw).hexdigest()
        output_info['frames'] = re.search(rb'Frames:\s*(\d+)', raw).group(1).decode() if re.search(rb'Frames:\s*(\d+)', raw) else None
        file.seek(0)
tasks.send_callback = capture_callback

async def main():
    async with httpx.AsyncClient(timeout=httpx.Timeout(settings.download_timeout_seconds), follow_redirects=True) as client:
        for run in range(2 if LABEL == 'repaired' else 1):
            stats.clear()
            counts.clear()
            source_hashes.clear()
            output_info.clear()
            start = time.perf_counter()
            await tasks.run_merge_task(client, settings, 'diagnostic-replay-1180', payload)
            elapsed = time.perf_counter() - start
            engine = mdm_transition._MDM_ENGINE
            print('BENCH_RESULT', json.dumps({'label': LABEL, 'run': run + 1, 'task': 1180, 'elapsed_seconds': round(elapsed, 3), 'device': str(engine.device) if engine else None, 'stages_seconds': {k: round(v, 3) for k,v in stats.items()}, 'counts': dict(counts), 'input_hashes': source_hashes, 'output': output_info}, ensure_ascii=False), flush=True)
            assert output_info.get('success'), 'Replay failed'

import re
asyncio.run(main())
'''
script = 'PAYLOAD = ' + repr(payload) + '\nLABEL = ' + repr(label) + '\n' + script
result = subprocess.run(['docker', 'exec', '-i', 'brss1-algorithm', '/app/.venv/bin/python', '-'], input=script, text=True, timeout=900)
raise SystemExit(result.returncode)
