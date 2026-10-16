"""Run on bdca via SSH stdin; change only algorithm GPU reservation."""
from pathlib import Path
from datetime import datetime
import shutil
import subprocess

p = Path('/brss1.0/docker-compose.yml')
s = p.read_text()
start = s.index('\n  algorithm:\n')
end = s.index('\n  # ========== MinIO', start)
block = s[start:end]
assert 'deploy:' not in block and 'gpus:' not in block, 'GPU configuration already present; inspect manually'
needle = '    restart: always\n'
assert block.count(needle) == 1
addition = '''    # Expose only host GPU 0 to the transition model.
    deploy:
      resources:
        reservations:
          devices:
            - driver: nvidia
              device_ids: ["0"]
              capabilities: [gpu]
'''
backup = p.with_name('docker-compose.yml.bak-gpu-' + datetime.now().strftime('%Y%m%d-%H%M%S'))
shutil.copy2(p, backup)
p.write_text(s[:start] + block.replace(needle, needle + addition, 1) + s[end:])
print('Backup:', backup, flush=True)
try:
    subprocess.run(['docker', 'compose', 'config', '--quiet'], cwd=p.parent, check=True)
except subprocess.CalledProcessError:
    shutil.copy2(backup, p)
    print('Validation failed; original compose restored', flush=True)
    raise
subprocess.run(['docker', 'compose', 'up', '-d', '--no-deps', 'algorithm'], cwd=p.parent, check=True)
subprocess.run(['docker', 'inspect', 'brss1-algorithm', '--format', 'DeviceRequests={{json .HostConfig.DeviceRequests}} Status={{.State.Status}}'], check=True)
