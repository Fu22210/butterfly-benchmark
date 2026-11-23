"""Validate the contest JSONL against its manifest and submission template."""
import json
import math
import sys
from pathlib import Path


def unique_keys(pairs: list) -> dict:
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f'Duplicate key {key}')
        result[key] = value
    return result


def read_rows(path: Path) -> dict:
    rows = {}
    for line in path.read_text(encoding='utf-8').splitlines():
        row = json.loads(line, object_pairs_hook=unique_keys)
        photo_id = row['photo_id']
        if type(photo_id) is not int or photo_id in rows:
            raise ValueError(f'{path}: invalid or repeated ID {photo_id}')
        rows[photo_id] = row
    if len(rows) != 400:
        raise ValueError(f'{path}: expected 400 rows, found {len(rows)}')
    return rows


def validate(manifest: Path, template: Path, submission: Path) -> None:
    if submission.stat().st_size > 5 * 1024 ** 2:
        raise ValueError('Submission exceeds 5 MiB')
    rows = read_rows(submission)
    if not (rows.keys() == read_rows(manifest).keys() == read_rows(template).keys()):
        raise ValueError('ID sets differ')
    counts = []
    for photo_id, row in rows.items():
        if row.keys() - {'photo_id', 'boxes', 'error', 'photo_url'} or row.get('error') is not None:
            raise ValueError(f'{photo_id}: invalid fields or model error')
        boxes = row['boxes']
        if not isinstance(boxes, list):
            raise ValueError(f'{photo_id}: expected a list of boxes')
        for box in boxes:
            if set(box) != {'left', 'top', 'width', 'height'}:
                raise ValueError(f'{photo_id}: invalid box fields')
            if any(type(v) not in (float, int) or not math.isfinite(v) or not 0 <= v <= 1 for v in box.values()):
                raise ValueError(f'{photo_id}: invalid coordinate')
            if (min(box['width'], box['height']) <= 0 or box['left'] + box['width'] > 1
                    or box['top'] + box['height'] > 1):
                raise ValueError(f'{photo_id}: invalid box extent')
        counts.append(len(boxes))
    print(f'{len(rows)} photos, {sum(counts)} boxes, {counts.count(0)} empty, '
          f'{sum(n > 1 for n in counts)} multi-object, {submission.stat().st_size} bytes')


if __name__ == '__main__':
    validate(*(Path(arg) for arg in sys.argv[1:]))
