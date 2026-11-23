"""Contest object F1: maximum-cardinality matching within each image, then micro-aggregation."""
import argparse
import json
import math
from pathlib import Path


def unique_keys(pairs):
    result = {}
    for k, v in pairs:
        if k in result:
            raise ValueError(f'Duplicate key {k}')
        result[k] = v
    return result


def load(path):
    data = {}
    for line in path.read_text().splitlines():
        r = json.loads(line, object_pairs_hook=unique_keys)
        i = r['photo_id']
        if type(i) is not int or i in data or r.get('error') is not None or not isinstance(r['boxes'], list):
            raise ValueError(f'Invalid row or duplicate photo ID {i}')
        rects = []
        for b in r['boxes']:
            if set(b) != {'left','top','width','height'}:
                raise ValueError(f'Invalid box keys in photo {i}')
            x, y, w, h = [b[k] for k in ('left', 'top', 'width', 'height')]
            if not all(type(v) in (float,int) and math.isfinite(v) and 0 <= v <= 1 for v in (x,y,w,h)):
                raise ValueError(f'Invalid coordinate in photo {i}')
            if w <= 0 or h <= 0 or x+w > 1+1e-12 or y+h > 1+1e-12:
                raise ValueError(f'Invalid box dimensions in photo {i}')
            rects.append((x,y,x+w,y+h))
        data[i] = rects
    return data


def iou(a,b):
    x,y,r,s = a
    X,Y,R,S = b
    inter = max(0,min(r,R)-max(x,X))*max(0,min(s,S)-max(y,Y))
    return inter/((r-x)*(s-y)+(R-X)*(S-Y)-inter)


def matches(gold, pred, threshold):
    edges = [[j for j,p in enumerate(pred) if iou(g,p)>=threshold] for g in gold]
    owner = {}
    def assign(i, seen):
        for j in edges[i]:
            if j in seen:
                continue
            seen.add(j)
            if j not in owner or assign(owner[j],seen):
                owner[j] = i
                return True
        return False
    return sum(assign(i,set()) for i in range(len(gold)))


def score(gold,pred,ids,threshold):
    tp = sum(matches(gold[i], pred[i], threshold) for i in ids)
    fp = sum(len(pred[i]) for i in ids)-tp
    fn = sum(len(gold[i]) for i in ids)-tp
    neg = [i for i in ids if not gold[i]]
    return dict(n=len(ids),tp=tp,fp=fp,fn=fn,
                precision=tp/(tp+fp) if tp+fp else 0,recall=tp/(tp+fn) if tp+fn else 0,
                f1=2*tp/(2*tp+fp+fn) if 2*tp+fp+fn else 0,
                fp_on_empty=sum(len(pred[i]) for i in neg),
                empty_with_fp=sum(bool(pred[i]) for i in neg),empty_images=len(neg))


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('gold',type=Path)
    parser.add_argument('out',type=Path)
    parser.add_argument('--metadata',type=Path)
    args=parser.parse_args()
    gold=load(args.gold)
    assert len(gold)==400
    groups={'all':list(gold)}
    if args.metadata:
        meta=list(map(json.loads,args.metadata.open()))
        assert {r['photo_id'] for r in meta}==gold.keys() and len(meta)==len(gold)
        for r in meta:
            groups.setdefault(r['source_category'],[]).append(r['photo_id'])
    report={}
    for name in ('dino','owl','qwen'):
        pred=load(args.out/f'{name}.jsonl')
        if pred.keys()!=gold.keys():
            raise ValueError(f'Wrong image set for {name}')
        history=list(map(json.loads,(args.out/f'{name}.journal.jsonl').open()))
        successful={r['photo_id']:r for r in history if r.get('error') is None}
        assert successful.keys()==gold.keys()
        report[name]={'seconds':sum(r['seconds'] for r in successful.values()),
                      'failed_attempts':sum(r.get('error') is not None for r in history),
                      'groups':{key:{str(t):score(gold,pred,ids,t) for t in (0.5,0.75)} for key,ids in groups.items()}}
    (args.out/'metrics.json').write_text(json.dumps(report,indent=2)+'\n')
    for name,r in report.items():
        print(name,r['groups']['all'],round(r['seconds'],2))


if __name__=='__main__':
    main()
