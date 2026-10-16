from pathlib import Path
from collections import Counter
p=Path('/tmp/brss1-merge-20261016-profile.raw')
counts=Counter()
total=0
for line in p.read_text().splitlines():
    stack,n=line.rsplit(' ',1)
    n=int(n)
    if 'merge_bvh_clips ' not in stack:
        continue
    total+=n
    if '_solve_local_rotations ' in stack:
        counts['CPU rotation solving']+=n
    elif 'export_transition_bvh ' in stack:
        counts['Other BVH export']+=n
    elif 'convert_bvh ' in stack:
        counts['BVH feature conversion']+=n
    elif 'generate (mdm/workbench/transitions.py:' in stack:
        counts['MDM generation (includes CPU preparation)']+=n
    else:
        counts['Other merge work']+=n
print('Merge thread samples:',total)
for k,n in counts.most_common():
    print(f'{k}: {n} ({100*n/total:.2f}%)')
rows=[]
for line in p.read_text().splitlines():
    stack,n=line.rsplit(' ',1)
    rows.append((int(n),stack))
print('Top 5 stacks:')
for n,stack in sorted(rows,reverse=True)[:5]:
    print(n,stack)
