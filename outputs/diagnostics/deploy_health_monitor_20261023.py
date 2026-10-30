"""Bounded deployment of four health-monitor files, preserving existing server code."""
import hashlib
from pathlib import Path
import shutil
import subprocess
import sys

root = Path('/brss1.0/services/algorithm/src/bvh_processing')
stage = Path('/brss1.0/backups/health-monitor-20261023-1208')
files = {
    'config.py': ('src/bvh_processing/config.py', 'dc6512aa79184412537fbfb1082d19e6f0a4a37d'),
    'main.py': ('src/bvh_processing/main.py', 'b2541435e1ee2f69dcd8898d2b7634adc4e55884'),
    'routes.py': ('src/bvh_processing/api/routes.py', '7d64d80cb6daab82104a423a8e0d6681f2b2a231'),
    'health.py': ('src/bvh_processing/services/health.py', None),
}
def check():
    for name, (relative, expected) in files.items():
        path = root / relative
        if expected is None:
            assert not path.exists(), f'New path already exists: {relative}'
        else:
            data = path.read_bytes()
            actual = hashlib.sha1(b'blob ' + str(len(data)).encode() + b'\0' + data).hexdigest()
            assert actual == expected, f'Server file changed: {relative}'

if sys.argv[1] == 'prepare':
    check()
    stage.mkdir(mode=0o700)
    (stage / 'incoming').mkdir()
    (stage / 'original').mkdir()
    for name, (relative, expected) in files.items():
        if expected:
            shutil.copy2(root / relative, stage / 'original' / name)
    print('BACKUP', stage)
elif sys.argv[1] == 'install':
    check()
    for name, (relative, _) in files.items():
        data = (stage / 'incoming' / name).read_bytes()
        compile(data, str(root / relative), 'exec')
    for name, (relative, _) in files.items():
        destination = root / relative
        temporary = destination.with_suffix('.py.health-update')
        shutil.copyfile(stage / 'incoming' / name, temporary)
        temporary.chmod(0o644)
        temporary.replace(destination)
    subprocess.run(['docker', 'restart', 'brss1-algorithm'], check=True)
    print('INSTALLED_HEALTH_MONITOR')
else:
    raise SystemExit('Unknown operation')
