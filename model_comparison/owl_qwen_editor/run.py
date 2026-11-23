"""Review each saved OWLv2 proposal in its full image, without reference annotations."""
import argparse
import json
import math
import re
import sys
import time
from pathlib import Path

import torch
from huggingface_hub import snapshot_download
from PIL import Image, ImageOps
from transformers import AutoProcessor

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from infer import MODELS, normalize

PROMPT = """Review ONE proposed bounding box in this photograph. The detector proposed {box} in normalized 0-1000 [x_min, y_min, x_max, y_max] coordinates.
Inspect the object at that box. If it is an adult butterfly or moth, edit the box to tightly enclose its visible wings, body, antennae, and legs. You may expand or shrink the box. Do not extend behind occlusions or outside the image. Keep good boundaries if already tight.
If the proposed object is a different insect, caterpillar, pupa, plant, shadow, or background, return []. Only review this candidate. Do not add other objects.
Return only a JSON array: either [{{"bbox_2d":[x_min,y_min,x_max,y_max]}}] in normalized 0-1000 coordinates of the full image, or []."""


def box1000(b: dict) -> list[int]:
    x, y = b['left'], b['top']
    return [round(1000*v) for v in (x, y, x+b['width'], y+b['height'])]


def review(model, processor, im: Image.Image, candidate: dict) -> str:
    text = PROMPT.format(box=json.dumps(box1000(candidate),separators=(',',':')))
    messages = [{'role':'user','content':[{'type':'image'},{'type':'text','text':text}]}]
    prompt = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = processor(text=[prompt], images=[im], return_tensors='pt').to('cuda')
    tokens = model.generate(**inputs, do_sample=False, max_new_tokens=512)
    return processor.decode(tokens[0,inputs.input_ids.shape[1]:], skip_special_tokens=True)


def parse(raw: str) -> list[dict]:
    s = re.sub(r'^\s*```(?:json)?\s*|\s*```\s*$', '', raw).strip()
    data = json.loads(s)
    if isinstance(data, dict):
        data = [data]
    elif isinstance(data, list) and len(data) == 4 and all(type(v) in (int, float) for v in data):
        data = [{'bbox_2d': data}]
    if not isinstance(data,list) or len(data)>1:
        raise ValueError('Expected zero or one proposed box')
    if not data:
        return []
    coords = data[0] if isinstance(data[0], list) else data[0]['bbox_2d']
    if any(type(v) not in (float,int) or not math.isfinite(v) or not 0 <= v <= 1000 for v in coords):
        raise ValueError('Invalid generated coordinate')
    return [normalize(coords,1000,1000)]


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    for key in ('images','owl_journal','out'):
        parser.add_argument(key,type=Path)
    args=parser.parse_args()
    paths={int(p.stem):p for p in args.images.iterdir() if p.is_file() and p.stem.isdecimal()}
    owl={}
    for line in args.owl_journal.read_text().splitlines():
        row=json.loads(line)
        if row['error'] is None:
            owl[row['photo_id']]=row
    if len(owl)!=400 or paths.keys()!=owl.keys():
        raise ValueError('Expected 400 matching originals and successful OWLv2 rows')
    args.out.mkdir(exist_ok=True,parents=True)
    log=args.out/'journal.jsonl'
    done={}
    if log.exists():
        for line in log.read_text().splitlines():
            row=json.loads(line)
            if row['error'] is None:
                done[row['photo_id']]=row
    repo,rev,cls=MODELS['qwen']
    snapshot=snapshot_download(repo,revision=rev,allow_patterns=['*.json','*.safetensors','*.txt','*.jinja'])
    processor=AutoProcessor.from_pretrained(snapshot,min_pixels=65536,max_pixels=1048576)
    model=cls.from_pretrained(snapshot,dtype=torch.bfloat16,attn_implementation='sdpa').eval().to('cuda')
    with log.open('a',buffering=1) as journal, torch.inference_mode():
        for n,i in enumerate(sorted(owl),1):
            if i in done:
                continue
            start=time.perf_counter()
            candidates=[b for b,s in zip(owl[i]['boxes'],owl[i]['scores'],strict=True) if s>=0.40]
            row=dict(photo_id=i,boxes=[],reviews=[],error=None)
            try:
                if candidates:
                    with Image.open(paths[i]) as src:
                        im=ImageOps.exif_transpose(src).convert('RGB')
                    for candidate in candidates:
                        record={'original':candidate}
                        row['reviews'].append(record)
                        record['raw']=review(model,processor,im,candidate)
                        record['edited']=parse(record['raw'])
                        row['boxes'].extend(record['edited'])
                    torch.cuda.synchronize()
                done[i]=row
            except Exception as exc:
                row.update(error=repr(exc), boxes=None)
                print(i,row['error'],flush=True)
            row['seconds']=time.perf_counter()-start
            journal.write(json.dumps(row,allow_nan=False)+'\n')
            if n%25==0:
                print(n,len(done),flush=True)
    if done.keys()!=owl.keys():
        raise SystemExit('Unresolved processing errors, see journal')
    target=args.out/'predictions.jsonl'
    with target.with_suffix('.tmp').open('w') as f:
        for i,row in sorted(done.items()):
            f.write(json.dumps({k:row[k] for k in ('photo_id','boxes','error')},allow_nan=False)+'\n')
    target.with_suffix('.tmp').replace(target)


if __name__=='__main__':
    main()
