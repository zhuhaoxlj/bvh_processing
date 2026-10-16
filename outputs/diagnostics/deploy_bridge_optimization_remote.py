from pathlib import Path
from datetime import datetime
import hashlib
import shutil
import subprocess
import py_compile

root=Path('/brss1.0/services/algorithm/src/bvh_processing')
relative=Path('src/bvh_processing/vendor/mdm/workbench/bvh_export.py')
dest=root/relative
staged=Path('/tmp/brss1-merge-optimization-20261016')/relative
expected='a8ad7e99fb9ac081572138b2e05d080ee84f2eb2801fcf71ea9b1236f54eff8a'
assert hashlib.sha256(dest.read_bytes()).hexdigest()==expected, 'Live source changed since inspection; stop deployment'
py_compile.compile(str(staged),doraise=True)
backup=dest.with_name(dest.name+'.bak-bridge-opt-'+datetime.now().strftime('%Y%m%d-%H%M%S'))
shutil.copy2(dest,backup)
temp=dest.with_name(dest.name+'.deploy-tmp')
shutil.copy2(staged,temp)
temp.replace(dest)
print('Backup:',backup,flush=True)
print('Deployed SHA256:',hashlib.sha256(dest.read_bytes()).hexdigest(),flush=True)
subprocess.run(['docker','compose','restart','algorithm'],cwd='/brss1.0',check=True)
print('Algorithm restarted; health verification required',flush=True)
