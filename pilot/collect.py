"""Collect a fixed CC0 proposal for human annotation; never generate gold labels."""
from __future__ import annotations

import argparse
import collections
import concurrent.futures
import hashlib
import html
import json
import random
import time
import urllib.parse
import urllib.request
from pathlib import Path

from PIL import Image, ImageDraw, ImageOps

ROOT = Path(__file__).resolve().parent
SEED = 20261115
GROUPS = {
    'day': {'taxon_id': 47224, 'term_id': 1, 'term_value_id': 2},
    'night': {'taxon_id': 47157, 'without_taxon_id': 47224, 'term_id': 1, 'term_value_id': 2},
    'larva': {'taxon_id': 47157, 'term_id': 1, 'term_value_id': 6},
    'pupa': {'taxon_id': 47157, 'term_id': 1, 'term_value_id': 4},
    'hymenoptera': {'taxon_id': 47201, 'term_id': 1, 'term_value_id': 2},
    'beetle': {'taxon_id': 47208, 'term_id': 1, 'term_value_id': 2},
    'plant': {'taxon_id': 47126},
}
LABELS = dict(day='Дневные бабочки', night='Ночные бабочки', larva='Гусеницы',
              pupa='Куколки', hymenoptera='Перепончатокрылые', beetle='Жуки', plant='Растения')
QUOTAS = {
    'main400': [180, 40, 15, 5, 50, 50, 60],
    'training30': [13, 3, 1, 0, 4, 4, 5],
    'bbox_exam20': [9, 2, 1, 0, 2, 3, 3],
    'review_exam20': [9, 2, 0, 1, 2, 3, 3],
    'control30': [14, 3, 2, 0, 4, 3, 4],
}
FIELDS = {
    'id': True, 'uri': True, 'observed_on': True, 'created_at': True, 'quality_grade': True,
    'user': {'id': True, 'login': True},
    'taxon': {'id': True, 'name': True, 'rank': True, 'ancestor_ids': True},
    'annotations': {'controlled_attribute_id': True, 'controlled_value_id': True},
    'photos': {'id': True, 'url': True, 'license_code': True, 'attribution': True},
}


def write_jsonl(path: Path, rows: list[dict]) -> None:
    path.write_text(''.join(json.dumps(x, ensure_ascii=False)+'\n' for x in rows), encoding='utf-8')


def read_jsonl(path: Path) -> list[dict]:
    return [json.loads(s) for s in path.read_text().splitlines() if s.strip()]


def get_bytes(url: str, retries: int = 3, timeout: int = 30) -> bytes:
    for attempt in range(retries):
        try:
            req = urllib.request.Request(url, headers={'User-Agent': 'ButterflyCoursePilot/1.0'})
            with urllib.request.urlopen(req, timeout=timeout) as response:
                return response.read()
        except (OSError, ValueError):
            if attempt == retries-1:
                raise
            time.sleep(2*(attempt+1))
    raise RuntimeError('unreachable')


def fetch_candidates() -> None:
    (ROOT/'cache').mkdir(exist_ok=True)
    excluded = {x['photo_id'] for x in read_jsonl(ROOT/'reference/contest_dataset.jsonl')}
    candidates: list[dict] = []
    for group, filters in GROUPS.items():
        # Distributed calendar windows, rather than a single burst of recent uploads.
        for year, month in [(2020, 3), (2021, 6), (2023, 9), (2025, 1)]:
            cache = ROOT/'cache'/f'{group}-{year}-{month:02}.json'
            params = dict(filters, photo_license='cc0', photos='true', per_page=200,
                          order_by='id', order='asc', created_d1=f'{year}-{month:02}-01',
                          created_d2=f'{year}-12-31', fields=json.dumps(FIELDS))
            url = 'https://api.inaturalist.org/v2/observations?'+urllib.parse.urlencode(params)
            if not cache.exists():
                cache.write_bytes(get_bytes(url))
                time.sleep(1.1)
            result = json.loads(cache.read_text())
            count = 0
            for obs in result['results']:
                photos = sorted(obs.get('photos', []), key=lambda p: p['id'])
                # One CC0 photograph per observation before sampling.
                photos = [p for p in photos if p.get('license_code') == 'cc0' and p['id'] not in excluded]
                if not photos or not obs.get('taxon'):
                    continue
                photo = random.Random(SEED+obs['id']).choice(photos)
                ext = urllib.parse.urlparse(photo['url']).path.rsplit('.', 1)[-1].lower()
                if ext not in ('jpg', 'jpeg', 'png'):
                    continue
                candidates.append(dict(
                    photo_id=photo['id'],
                    photo_url=f"https://inaturalist-open-data.s3.amazonaws.com/photos/{photo['id']}/original.{ext}",
                    observation_id=obs['id'], observation_url=obs.get('uri') or f"https://www.inaturalist.org/observations/{obs['id']}",
                    photo_page_url=f"https://www.inaturalist.org/photos/{photo['id']}",
                    author=obs['user']['login'], author_id=obs['user']['id'],
                    license='cc0', attribution=photo.get('attribution'), source_group=group,
                    taxon_id=obs['taxon']['id'], taxon_name=obs['taxon']['name'],
                    ancestor_ids=obs['taxon']['ancestor_ids'], annotations=obs.get('annotations', []),
                    observed_on=obs.get('observed_on'), created_at=obs.get('created_at'),
                    query_year=year, query_url=url, quality_grade=obs.get('quality_grade'),
                    category_status='source_metadata_not_verified_per_photo',
                ))
                count += 1
            print(f'candidates {group} {year}: {count}', flush=True)
    write_jsonl(ROOT/'candidates.jsonl', candidates)
    (ROOT/'sampling_config.json').write_text(json.dumps(dict(seed=SEED, groups=GROUPS,
        quotas=QUOTAS, max_per_author=12, max_per_taxon=6, near_duplicate_dhash_distance=3), indent=2))


def dhash(im: Image.Image) -> int:
    pixels = list(im.convert('L').resize((9, 8), Image.Resampling.LANCZOS).get_flattened_data())
    return sum(int(pixels[y*9+x] > pixels[y*9+x+1]) << (y*8+x) for y in range(8) for x in range(8))


def download(row: dict) -> tuple[dict, str | None]:
    row = dict(row)
    extension = row['photo_url'].rsplit('.', 1)[-1]
    path = ROOT/'images'/f"{row['photo_id']}.{extension}"
    try:
        if not path.exists():
            data = get_bytes(row['photo_url'], retries=2, timeout=15)
            path.write_bytes(data)
        with Image.open(path) as raw:
            raw.load()
            width, height = raw.size
            # Originals retain their exact bytes. Flag orientation for downstream bbox code.
            orientation = raw.getexif().get(274, 1)
            if orientation != 1:
                return row, 'nontrivial EXIF rotation: excluded to keep browser/model coordinates consistent'
            im = ImageOps.exif_transpose(raw).convert('RGB')
            if min(im.size) < 200:
                return row, 'image smaller than 200 pixels on its shorter side'
            fingerprint = dhash(im)
            thumb = ImageOps.contain(im, (640, 640))
            thumb.save(ROOT/'previews'/f"{row['photo_id']}.jpg", quality=87)
        row.update(local_path=str(path.relative_to(ROOT)), width=width, height=height,
                   display_width=im.width, display_height=im.height, exif_orientation=orientation,
                   sha256=hashlib.sha256(path.read_bytes()).hexdigest(), dhash=f'{fingerprint:016x}')
        return row, None
    except (OSError, ValueError) as exc:
        # Failed/partial downloads must never be mistaken for valid empty images.
        path.unlink(missing_ok=True)
        return row, f'{type(exc).__name__}: {exc}'


def select() -> None:
    for name in ('images', 'previews', 'manifests', 'contact_sheets'):
        (ROOT/name).mkdir(exist_ok=True)
    candidates = read_jsonl(ROOT/'candidates.jsonl')
    rng = random.Random(SEED)
    rng.shuffle(candidates)
    excluded_obs = set(json.loads((ROOT/'reference/exclusions.json').read_text())['observation_ids'])
    exam_hashes = json.loads((ROOT/'reference/exclusions.json').read_text())['dhashes']
    hashes = [int(x, 16) for x in exam_hashes]
    totals = {g: sum(q[i] for q in QUOTAS.values()) for i, g in enumerate(GROUPS)}
    authors: collections.Counter = collections.Counter()
    taxa: collections.Counter = collections.Counter()
    seen_obs: set[int] = set(excluded_obs)
    seen_ids: set[int] = set()
    sha: set[str] = set()
    selected: list[dict] = []
    rejected: list[dict] = []
    with concurrent.futures.ThreadPoolExecutor(max_workers=16) as pool:
        for group, target in totals.items():
            queue = iter(r for r in candidates if r['source_group'] == group)
            chosen: list[dict] = []
            while len(chosen) < target:
                batch = []
                for row in queue:
                    if row['observation_id'] in seen_obs or row['photo_id'] in seen_ids:
                        continue
                    if authors[row['author_id']] >= 12 or taxa[row['taxon_id']] >= 6:
                        continue
                    batch.append(row)
                    if len(batch) >= min(32, target-len(chosen)):
                        break
                if not batch:
                    raise RuntimeError(f'Insufficient diverse candidates for {group}: {len(chosen)}/{target}')
                for row, error in pool.map(download, batch):
                    reason = error
                    value = int(row.get('dhash', '0'), 16)
                    if not reason and row['observation_id'] in seen_obs:
                        reason = 'repeated observation'
                    if not reason and (authors[row['author_id']] >= 12 or taxa[row['taxon_id']] >= 6):
                        reason = 'author/taxon diversity cap'
                    if not reason and (row['sha256'] in sha or any((value ^ h).bit_count() <= 3 for h in hashes)):
                        reason = 'exact or near duplicate of selected/contest photograph'
                    if reason:
                        rejected.append(dict(photo_id=row['photo_id'], reason=reason))
                        continue
                    chosen.append(row)
                    authors[row['author_id']] += 1
                    taxa[row['taxon_id']] += 1
                    seen_obs.add(row['observation_id'])
                    seen_ids.add(row['photo_id'])
                    sha.add(row['sha256'])
                    hashes.append(value)
                print(f'selected {group}: {len(chosen)}/{target}', flush=True)
            rng.shuffle(chosen)
            cursor = 0
            i = list(GROUPS).index(group)
            for split, quotas in QUOTAS.items():
                n = quotas[i]
                selected.extend(dict(row, split=split) for row in chosen[cursor:cursor+n])
                cursor += n
            write_jsonl(ROOT/'selection_checkpoint.jsonl', selected)
            write_jsonl(ROOT/'rejections.jsonl', rejected)
    for split in QUOTAS:
        rows = [r for r in selected if r['split'] == split]
        rng.shuffle(rows)
        write_jsonl(ROOT/'manifests'/f'{split}.jsonl', rows)
        write_jsonl(ROOT/'manifests'/f'{split}_images_only.jsonl', [dict(photo_id=r['photo_id'],photo_url=r['photo_url']) for r in rows])
    write_jsonl(ROOT/'manifests/all500.jsonl', selected)
    # No fabricated bbox gold: pending records are intentionally not valid contest answers.
    write_jsonl(ROOT/'manifests/staff100_gold_PENDING.jsonl', [dict(photo_id=r['photo_id'],
        split=r['split'], boxes=None, status='needs_independent_human_annotation') for r in selected if r['split'] != 'main400'])
    write_jsonl(ROOT/'manifests/main400_gold_PENDING.jsonl', [dict(photo_id=r['photo_id'], boxes=None,
        status='awaiting_bbox_and_review') for r in selected if r['split'] == 'main400'])
    build_gallery(selected)
    validate()


def build_gallery(rows: list[dict]) -> None:
    sections = []
    notes_path = ROOT/'visual_review_notes.json'
    notes = {r['photo_id']: r['note'] for r in json.loads(notes_path.read_text())} if notes_path.exists() else {}
    for split in QUOTAS:
        subset = [r for r in rows if r['split'] == split]
        cards = []
        for r in subset:
            esc = html.escape
            note = f'<p style="background:#fff1cd;padding:8px">Заметка при просмотре: {esc(notes[r["photo_id"]])}</p>' if r['photo_id'] in notes else ''
            cards.append(f'<article data-group="{r["source_group"]}"><a href="{esc(r["local_path"])}"><img loading="lazy" src="previews/{r["photo_id"]}.jpg"></a><b>{r["photo_id"]} · {LABELS[r["source_group"]]}</b><p>{esc(r["taxon_name"])}<br>Пользователь наблюдения: {esc(r["author"])} · CC0</p>{note}<a href="{esc(r["observation_url"])}" target="_blank">Наблюдение</a> · <a href="{esc(r["photo_url"])}" target="_blank">Оригинал</a></article>')
        sections.append(f'<section data-split="{split}"><h2>{split} ({len(subset)})</h2><div class="grid">'+''.join(cards)+'</div></section>')
        for start in range(0, len(subset), 25):
            sheet = Image.new('RGB', (1250, 1100), 'white')
            draw = ImageDraw.Draw(sheet)
            for i, r in enumerate(subset[start:start+25]):
                x, y = i % 5 * 250, i // 5 * 220
                with Image.open(ROOT/'previews'/f"{r['photo_id']}.jpg") as im:
                    im = ImageOps.contain(im, (246, 190))
                    sheet.paste(im, (x+(250-im.width)//2, y))
                draw.text((x+3,y+193), f"{r['photo_id']}  {r['source_group']}", fill='black')
            sheet.save(ROOT/'contact_sheets'/f'{split}-{start//25+1:02}.jpg', quality=88)
    controls = '<select id="split"><option value="">Все части</option>'+''.join(f'<option>{s}</option>' for s in QUOTAS)+'</select>'
    controls += '<select id="group"><option value="">Все группы</option>'+''.join(f'<option value="{g}">{label}</option>' for g,label in LABELS.items())+'</select>'
    page = '<!doctype html><html lang="ru"><meta charset="utf-8"><title>Бабочки: выборка 400 + 100</title><style>body{font:16px system-ui;margin:30px;background:#f4f5f1;color:#18241e}h1{font-size:30px}.grid{display:grid;grid-template-columns:repeat(auto-fill,minmax(260px,1fr));gap:16px}article{background:white;padding:12px;border-radius:8px}img{width:100%;height:220px;object-fit:contain}b{display:block}select{padding:10px;margin:10px}a{color:#285c4c}</style><h1>Предложенная выборка: 400 + 100</h1><p>Категории получены из наблюдений iNaturalist. Это не эталонная разметка. Рамки и отсутствие взрослых бабочек ещё требуют проверки людьми.</p>'+controls+''.join(sections)+'''<script>
function filter(){document.querySelectorAll('section').forEach(s=>s.hidden=!!split.value&&s.dataset.split!==split.value);document.querySelectorAll('article').forEach(c=>c.hidden=!!group.value&&c.dataset.group!==group.value)}
const split=document.getElementById('split'),group=document.getElementById('group');split.onchange=group.onchange=filter;
</script></html>'''
    (ROOT/'gallery.html').write_text(page, encoding='utf-8')


def validate() -> None:
    rows = read_jsonl(ROOT/'manifests/all500.jsonl')
    exam = read_jsonl(ROOT/'reference/contest_dataset.jsonl')
    assert len(rows) == 500
    assert len({r['photo_id'] for r in rows}) == 500
    assert len({r['observation_id'] for r in rows}) == 500
    assert not ({r['photo_id'] for r in rows} & {r['photo_id'] for r in exam})
    exclusion_data = json.loads((ROOT/'reference/exclusions.json').read_text())
    assert not ({r['observation_id'] for r in rows} & set(exclusion_data['observation_ids']))
    assert all(r['license'] == 'cc0' for r in rows)
    assert len({r['sha256'] for r in rows}) == 500
    assert all(r['exif_orientation'] == 1 for r in rows)
    assert max(collections.Counter(r['author_id'] for r in rows).values()) <= 12
    assert max(collections.Counter(r['taxon_id'] for r in rows).values()) <= 6
    fingerprints = [int(h, 16) for h in json.loads((ROOT/'reference/exclusions.json').read_text())['dhashes']]
    for row in rows:
        h = int(row['dhash'], 16)
        assert all((h ^ prior).bit_count() > 3 for prior in fingerprints)
        fingerprints.append(h)
        expected = GROUPS[row['source_group']]
        assert expected['taxon_id'] in row['ancestor_ids']
        if 'without_taxon_id' in expected:
            assert expected['without_taxon_id'] not in row['ancestor_ids']
        if 'term_value_id' in expected:
            assert any(a['controlled_attribute_id'] == 1 and a['controlled_value_id'] == expected['term_value_id'] for a in row['annotations'])
    for split, quotas in QUOTAS.items():
        actual = collections.Counter(r['source_group'] for r in rows if r['split'] == split)
        assert all(actual[g] == n for g,n in zip(GROUPS,quotas))
        public_inputs = read_jsonl(ROOT/'manifests'/f'{split}_images_only.jsonl')
        assert len(public_inputs) == sum(quotas)
        assert all(set(r) == {'photo_id', 'photo_url'} for r in public_inputs)
        assert {r['photo_id'] for r in public_inputs} == {r['photo_id'] for r in rows if r['split'] == split}
    for r in rows:
        assert hashlib.sha256((ROOT/r['local_path']).read_bytes()).hexdigest() == r['sha256']
    summary = dict(total=500, split_counts=dict(collections.Counter(r['split'] for r in rows)),
        group_counts=dict(collections.Counter(r['source_group'] for r in rows)),
        authors=len({r['author_id'] for r in rows}), taxa=len({r['taxon_id'] for r in rows}),
        max_photos_per_author=max(collections.Counter(r['author_id'] for r in rows).values()),
        max_photos_per_taxon=max(collections.Counter(r['taxon_id'] for r in rows).values()),
        bytes=sum((ROOT/r['local_path']).stat().st_size for r in rows),
        exif_orientations=dict(collections.Counter(r['exif_orientation'] for r in rows)),
        status='proposal_pending_human_bbox_and_review', seed=SEED)
    (ROOT/'validation.json').write_text(json.dumps(summary, ensure_ascii=False, indent=2))
    print(json.dumps(summary,ensure_ascii=False,indent=2),flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('action', choices=['fetch', 'select', 'validate'])
    args = parser.parse_args()
    {'fetch': fetch_candidates, 'select': select, 'validate': validate}[args.action]()
