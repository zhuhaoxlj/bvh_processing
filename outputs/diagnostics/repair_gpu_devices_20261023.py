"""Apply NVIDIA's explicit-device mitigation to algorithm only; run via SSH stdin."""
from pathlib import Path
from datetime import datetime
import shutil
import subprocess

p = Path('/brss1.0/docker-compose.yml')
s = p.read_text()
start = s.index('\n  algorithm:\n')
end = s.find('\n  # ========== MinIO', start)
assert end > start, 'Algorithm block boundary not found'
block = s[start:end]
assert 'device_ids:' in block and 'driver: nvidia' in block, 'Existing GPU reservation missing'
assert '\n    devices:\n' not in block, 'Explicit devices already configured; inspect first'
devices = ['/dev/nvidia0', '/dev/nvidiactl', '/dev/nvidia-uvm', '/dev/nvidia-uvm-tools']
for device in devices:
    assert Path(device).is_char_device(), f'Device missing: {device}'
needle = '    restart: always\n'
assert block.count(needle) == 1
addition = '    # Explicit GPU nodes keep device access visible to runc/cgroup updates.\n    devices:\n' + ''.join(f'      - {device}:{device}\n' for device in devices)
backup = p.with_name('docker-compose.yml.bak-gpu-devices-' + datetime.now().strftime('%Y%m%d-%H%M%S'))
shutil.copy2(p, backup)
p.write_text(s[:start] + block.replace(needle, needle + addition, 1) + s[end:])
print('BACKUP', backup, flush=True)
try:
    subprocess.run(['docker', 'compose', 'config', '--quiet'], cwd=p.parent, check=True)
except subprocess.CalledProcessError:
    shutil.copy2(backup, p)
    print('Compose validation failed; original restored', flush=True)
    raise
subprocess.run(['docker', 'compose', 'up', '-d', '--no-deps', '--force-recreate', 'algorithm'], cwd=p.parent, check=True)
subprocess.run(['docker', 'inspect', 'brss1-algorithm', '--format', 'Devices={{json .HostConfig.Devices}} GPU={{json .HostConfig.DeviceRequests}}'], check=True)
