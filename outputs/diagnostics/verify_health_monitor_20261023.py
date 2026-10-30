"""Run inside algorithm container to verify two periodic probes and HTTP cache latency."""
import json
import time
import urllib.error
import urllib.request

base = 'http://127.0.0.1:9001'
deadline = time.monotonic() + 180
initial_count = None
last_count = None
latencies = []
while time.monotonic() < deadline:
    try:
        start = time.perf_counter()
        with urllib.request.urlopen(base + '/health/details', timeout=3) as response:
            snapshot = json.load(response)
        latencies.append(time.perf_counter() - start)
        count = snapshot['gpu']['check_count']
        if initial_count is None:
            initial_count = count
        if count != last_count:
            print('HEALTH_SNAPSHOT', json.dumps(snapshot, ensure_ascii=False), flush=True)
            last_count = count
        with urllib.request.urlopen(base + '/health', timeout=3) as response:
            assert json.load(response) == {'status': 'ok'}
        if count >= max(2, initial_count + 1):
            assert snapshot['gpu']['compute_ok'] and not snapshot['gpu']['stale']
            assert snapshot['ready']
            with urllib.request.urlopen(base + '/health/ready', timeout=3) as response:
                assert response.status == 200
            print('VERIFIED_PERIODIC_CHECKS', count, 'max_details_latency_ms', round(max(latencies) * 1000, 3), flush=True)
            break
    except urllib.error.HTTPError as error:
        raise RuntimeError(f'Health endpoint HTTP {error.code}') from None
    except urllib.error.URLError:
        pass
    time.sleep(2)
else:
    raise SystemExit('Timed out waiting for successful periodic GPU checks')
