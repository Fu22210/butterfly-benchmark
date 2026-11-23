"""Run one fixed configuration on the supplied directory of photo_id.jpg originals."""
import argparse
import json
import math
import re
import time
from pathlib import Path

import torch
from huggingface_hub import snapshot_download
from PIL import Image, ImageOps
from torchvision.ops import nms
from transformers import (
    AutoProcessor, GroundingDinoForObjectDetection, Owlv2ForObjectDetection,
    Qwen3VLForConditionalGeneration,
)

MODELS = {
    'dino': ('IDEA-Research/grounding-dino-tiny', 'a2bb814dd30d776dcf7e30523b00659f4f141c71', GroundingDinoForObjectDetection),
    'owl': ('google/owlv2-base-patch16', '2a1560802f8cf3c408fec9b809d705f56a2f7146', Owlv2ForObjectDetection),
    'qwen': ('Qwen/Qwen3-VL-2B-Instruct', '89644892e4d85e24eaac8bacfd4f463576704203', Qwen3VLForConditionalGeneration),
}
TEXT = {
    'dino': 'butterfly. moth.',
    'owl': ['a photo of a butterfly', 'a photo of a moth'],
    'qwen': ('Find every adult butterfly or moth. Draw one tight bounding box per individual around '
             'its visible wings, body, antennae, and legs. Exclude caterpillars, pupae, other insects, and shadows. '
             'Return only a JSON array of objects with a "bbox_2d" field [x1,y1,x2,y2] '
             'in normalized 0-1000 coordinates. If none are present, return [].'),
}


def normalize(coords: list, width: int, height: int) -> dict:
    if len(coords) != 4 or any(type(v) not in (int, float) or not math.isfinite(v) for v in coords):
        raise ValueError(f'Invalid box: {coords}')
    x, y, right, bottom = [min(1., max(0., v / scale)) for v, scale in zip(coords, (width, height) * 2)]
    if right <= x or bottom <= y:
        raise ValueError(f'Empty box: {coords}')
    return dict(left=x, top=y, width=right-x, height=bottom-y)


def detect(name, model, processor, im):
    text = TEXT[name] if name == 'dino' else [TEXT[name]]
    inputs = processor(images=im, text=text, return_tensors='pt').to('cuda')
    outputs = model(**inputs)
    sizes = [(im.height, im.width)]
    if name == 'dino':
        det = processor.post_process_grounded_object_detection(
            outputs, inputs.input_ids, threshold=0.30, text_threshold=0.25, target_sizes=sizes)[0]
    else:
        det = processor.post_process_grounded_object_detection(outputs, threshold=0.10, target_sizes=sizes)[0]
    boxes, scores = det['boxes'].float(), det['scores'].float()
    boxes[:, 0::2].clamp_(0, im.width)
    boxes[:, 1::2].clamp_(0, im.height)
    valid = (boxes[:, 2:] > boxes[:, :2]).all(dim=1)
    boxes, scores = boxes[valid], scores[valid]
    keep = nms(boxes, scores, 0.50)
    return [normalize(b, im.width, im.height) for b in boxes[keep].tolist()], {'scores': scores[keep].tolist()}


def generate(model, processor, im):
    messages = [{'role': 'user', 'content': [{'type':'image'}, {'type':'text','text':TEXT['qwen']}]}]
    prompt = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = processor(text=[prompt], images=[im], return_tensors='pt').to('cuda')
    tokens = model.generate(**inputs, max_new_tokens=512, do_sample=False)
    raw = processor.decode(tokens[0,inputs.input_ids.shape[1]:], skip_special_tokens=True)
    string = re.sub(r'^\s*```(?:json)?\s*|\s*```\s*$', '', raw).strip()
    try:
        rows = json.loads(string)
        if not isinstance(rows, list):
            raise ValueError('Expected JSON array')
        boxes = [normalize(r['bbox_2d'], 1000, 1000) for r in rows]
    except (ValueError, KeyError, TypeError) as exc:
        raise ValueError(f'{exc}; generated={raw!r}') from exc
    return boxes, {'text': raw}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('name', choices=MODELS)
    parser.add_argument('images', type=Path)
    parser.add_argument('out', type=Path)
    parser.add_argument('--limit', type=int)
    args = parser.parse_args()
    files = sorted((p for p in args.images.iterdir() if p.is_file() and p.stem.isdecimal()),
                   key=lambda p: int(p.stem))
    if len({int(p.stem) for p in files}) != len(files):
        raise ValueError('Duplicate photo_id among image files')
    if args.limit:
        files = files[:args.limit]
    elif len(files) != 400:
        raise ValueError(f'Expected 400 photos, got {len(files)}')
    args.out.mkdir(exist_ok=True, parents=True)
    journal = args.out / f'{args.name}.journal.jsonl'
    cached = {r['photo_id']:r for r in map(json.loads,journal.open()) if r['error'] is None} if journal.exists() else {}
    repo, rev, cls = MODELS[args.name]
    options = dict(min_pixels=65536, max_pixels=1048576) if args.name == 'qwen' else {}
    snapshot = snapshot_download(repo, revision=rev, allow_patterns=['*.json','*.safetensors','*.txt','*.jinja'])
    processor = AutoProcessor.from_pretrained(snapshot, **options)
    load = dict(dtype=torch.bfloat16, attn_implementation='sdpa') if args.name == 'qwen' else dict(dtype=torch.float32)
    model = cls.from_pretrained(snapshot, **load).eval().to('cuda')
    with torch.inference_mode(), journal.open('a', buffering=1) as log:
        for n, path in enumerate(files,1):
            photo_id = int(path.stem)
            if photo_id in cached:
                continue
            start = time.perf_counter()
            try:
                with Image.open(path) as src:
                    im = ImageOps.exif_transpose(src).convert('RGB')
                if args.name == 'qwen':
                    boxes, extra = generate(model, processor, im)
                else:
                    boxes, extra = detect(args.name, model, processor, im)
                row = dict(photo_id=photo_id, boxes=boxes, error=None, **extra)
                cached[photo_id] = row
            except Exception as exc:
                row = dict(photo_id=photo_id, boxes=None, error=repr(exc))
                print(photo_id, row['error'], flush=True)
                if n == 1:
                    raise
            torch.cuda.synchronize()
            row['seconds'] = time.perf_counter()-start
            log.write(json.dumps(row, allow_nan=False)+'\n')
            if n % 25 == 0:
                print(args.name, n, len(cached), flush=True)
    if any(int(p.stem) not in cached for p in files):
        raise SystemExit('Unresolved errors in journal')
    with (args.out / f'{args.name}.jsonl').open('w') as f:
        for p in files:
            r = cached[int(p.stem)]
            f.write(json.dumps({k:r[k] for k in ('photo_id','boxes','error')})+'\n')


if __name__ == '__main__':
    main()
