"""Independent brute-force verification for this gold set (0, 1 or 2 objects)."""
import numpy as np
import json
import itertools
from pathlib import Path
from hashlib import sha256
wd=Path(__file__).resolve().parent
root=wd/'predictions'
gold={r['photo_id']:r['boxes'] for r in map(json.loads,(wd/'gold.jsonl').open())}
ids=sorted(gold)
assert max(map(len,gold.values()))==2 and sum(map(len,gold.values()))==193
metrics=json.load((root/'metrics.json').open())
def interval_overlap(a,b,start,length):
    return max(0,min(a[start]+a[length],b[start]+b[length])-max(a[start],b[start]))
def exhaustive(g,p,t):
    valid=[]
    for a in g:
        links=[]
        for j,b in enumerate(p):
            inter=interval_overlap(a,b,'left','width')*interval_overlap(a,b,'top','height')
            if inter >= t*(a['width']*a['height']+b['width']*b['height']-inter):
                links.append(j)
        valid.append(links)
    if len(g)==2 and any(i!=j for i,j in itertools.product(*valid)):
        return 2
    return int(any(valid))
for n in ['dino','owl','qwen']:
    rows=list(map(json.loads,(root/f'{n}.jsonl').open()))
    assert all(set(r)=={'photo_id','boxes','error'} and r['error'] is None for r in rows)
    p={r['photo_id']:r['boxes'] for r in rows}
    assert len(rows)==400 and p.keys()==gold.keys()
    for t in [.5,.75]:
        tp=sum(exhaustive(gold[i],p[i],t) for i in ids)
        fp=sum(map(len,p.values()))-tp
        fn=193-tp
        assert (tp,fp,fn)==tuple(metrics[n]['groups']['all'][str(t)][k] for k in ['tp','fp','fn'])
    print(n,'independent count OK',sha256((root/f'{n}.jsonl').read_bytes()).hexdigest())
