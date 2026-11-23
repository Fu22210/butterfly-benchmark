"""Score the editor and paired controls using the project's maximum-cardinality matcher."""
import argparse
import json
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from evaluate import iou, load, matches, score


def rect(b: dict) -> tuple:
    x,y=b['left'],b['top']
    return (x,y,x+b['width'],y+b['height'])


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('project', type=Path)
    parser.add_argument('results', type=Path)
    args=parser.parse_args()
    root,out=args.project,args.results
    g=load(root/'gold.jsonl')
    edited=load(out/'predictions.jsonl')
    journal=list(map(json.loads,(out/'journal.jsonl').open()))
    owl={r['photo_id']:r for r in map(json.loads,(root/'predictions/owl.journal.jsonl').open()) if r['error'] is None}
    reviewed={r['photo_id']:r for r in journal if r['error'] is None}
    assert len(g)==400 and g.keys()==reviewed.keys()==owl.keys()==edited.keys()
    full,kept={},{}
    quality=Counter()
    changes=[]
    examples=[]
    for i in sorted(g):
        rs=reviewed[i]['reviews']
        refs=[b for b,s in zip(owl[i]['boxes'],owl[i]['scores'],strict=True) if s>=0.4]
        assert refs==[r['original'] for r in rs]
        assert edited[i]==[rect(b) for r in rs for b in r['edited']]
        full[i]=[rect(b) for b in refs]
        kept[i]=[rect(r['original']) for r in rs if r['edited']]
        for r in rs:
            if not r['edited']:
                continue
            before=rect(r['original']); after=rect(r['edited'][0])
            to_grid=lambda b: tuple(round(x*1000) for x in b)
            quality['changed_at_1000']+=to_grid(before)!=to_grid(after)
            if not g[i]:
                continue
            a=max(iou(before,b) for b in g[i])
            b=max(iou(after,t) for t in g[i])
            changes.append(b-a)
            quality['improved_001' if b-a>=0.01 else 'worse_001' if a-b>=0.01 else 'within_001']+=1
            examples.append((i,a,b))
    # For this control, only edit geometry: rejected proposals keep their original box.
    edit_only = {i:[rect(r['edited'][0]) if r['edited'] else rect(r['original'])
                    for r in reviewed[i]['reviews']] for i in g}
    predictions={'owl':full,'accepted_original':kept,'edit_only':edit_only,'edited':edited,
                 'previous':load(root/'strategy_results/team_predictions.jsonl')}
    metrics={k:{str(t):score(g,b,sorted(g),t) for t in (0.5,0.75)} for k,b in predictions.items()}
    transitions={}
    for t in (0.5,0.75):
        diff=[matches(g[i],edited[i],t)-matches(g[i],kept[i],t) for i in g]
        transitions[str(t)]={'tp_gained':sum(max(d,0) for d in diff),'tp_lost':sum(max(-d,0) for d in diff)}
    summary=dict(metrics=metrics,candidates=sum(map(len,full.values())),proposed_images=sum(map(bool,full.values())),
                 accepted=sum(map(len,kept.values())),accepted_images=sum(map(bool,kept.values())),
                 candidate_images_without_gold=sum(bool(full[i]) and not g[i] for i in g),
                 gold_on_no_candidate_images=sum(len(g[i]) for i in g if not full[i]),
                 geometry=quality, transitions=transitions, mean_best_iou_change=sum(changes)/len(changes) if changes else None,
                 examples_worse=sorted(examples,key=lambda v:v[2]-v[1])[:3],
                 examples_better=sorted(examples,key=lambda v:v[2]-v[1],reverse=True)[:3],
                 failed_attempts=sum(r['error'] is not None for r in journal),
                 owl_seconds=sum(r['seconds'] for r in owl.values()),
                 editor_seconds=sum(r['seconds'] for r in journal))
    (out/'metrics.json').write_text(json.dumps(summary,indent=2)+'\n')
    print(json.dumps(summary,indent=2))


if __name__=='__main__':
    main()
