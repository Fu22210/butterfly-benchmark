"""Download the contest manifest verbatim, validating each original before inference."""
import json
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from urllib.request import urlopen

from PIL import Image


def download(row: dict) -> None:
    url = row['photo_url']
    photo_id = row['photo_id']
    prefix = f'https://inaturalist-open-data.s3.amazonaws.com/photos/{photo_id}/original.'
    if not url.startswith(prefix) or not url[len(prefix):].isalnum():
        raise ValueError(f'Unexpected original URL for {photo_id}')
    path = output_dir / f'{photo_id}.{url[len(prefix):]}'
    for attempt in range(4):
        try:
            if not path.exists():
                with urlopen(url, timeout=60) as source:
                    data = source.read()
                path.with_suffix('.part').write_bytes(data)
                with Image.open(path.with_suffix('.part')) as image:
                    image.load()
                path.with_suffix('.part').replace(path)
            return
        except Exception as exc:
            print(photo_id, attempt + 1, repr(exc), flush=True)
            if attempt == 3:
                raise
            time.sleep(2 ** attempt)


if __name__ == '__main__':
    rows = [json.loads(line) for line in Path(sys.argv[1]).read_text().splitlines()]
    if len(rows) != 400 or len({row['photo_id'] for row in rows}) != 400:
        raise ValueError('Expected 400 unique IDs')
    output_dir = Path(sys.argv[2])
    output_dir.mkdir(parents=True, exist_ok=True)
    with ThreadPoolExecutor(max_workers=8) as pool:
        list(pool.map(download, rows))
    print('Downloaded 400 originals', flush=True)
