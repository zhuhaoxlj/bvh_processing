import sys
from pathlib import Path
from time import perf_counter
import numpy as np
sys.path.insert(0, '/tmp/brss1-merge-optimization-20261016/tests')
from test_mdm_merge import FakeEngine, _clips, source_bvh
from bvh_processing.vendor.mdm.merge import merge_bvh_clips
from bvh_processing.vendor.mdm.workbench import bvh_export

source=source_bvh.__wrapped__()
solve=bvh_export._solve_local_rotations
positions=bvh_export._target_positions
latest=[]

def capture(*args, **kwargs):
    target=positions(*args, **kwargs)
    latest[:] = [target]
    return target
bvh_export._target_positions=capture
results={}
for mode in ('legacy','optimized'):
    stats={'frames':0,'solve_seconds':0.0}
    def measured(target, template):
        t=perf_counter()
        if mode=='legacy':
            full=latest[0]
            start=(target.ctypes.data-full.ctypes.data)//full.strides[0]
            stats['frames']+=len(full)
            result=solve(full,template)[start:start+len(target)]
        else:
            stats['frames']+=len(target)
            result=solve(target,template)
        stats['solve_seconds']+=perf_counter()-t
        return result
    bvh_export._solve_local_rotations=measured
    t=perf_counter()
    result=merge_bvh_clips(_clips(source, 12), [0.5]*11, engine=FakeEngine(),seed=21,scale=0.01,up_axis='Y')
    stats['total_seconds']=perf_counter()-t
    results[mode]=(result,stats)
    print(mode,stats,flush=True)
assert results['legacy'][0]==results['optimized'][0]
print('Exact BVH byte equality: PASS')
print('Total speedup:',results['legacy'][1]['total_seconds']/results['optimized'][1]['total_seconds'])
