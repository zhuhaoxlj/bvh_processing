from pathlib import Path
import shutil
root=Path('/brss1.0/services/algorithm/src/bvh_processing')
stage=Path('/tmp/brss1-merge-optimization-20261016')
assert not stage.exists(), 'Stage already exists'
stage.mkdir()
shutil.copytree(root/'src', stage/'src', ignore=shutil.ignore_patterns('resources','__pycache__'))
(stage/'src/bvh_processing/resources').symlink_to(root/'src/bvh_processing/resources', target_is_directory=True)
shutil.copytree(root/'tests', stage/'tests', ignore=shutil.ignore_patterns('__pycache__'))
shutil.copy2(root/'src/bvh_processing/vendor/mdm/workbench/bvh_export.py', stage/'bvh_export.original.py')
print(stage)
