"""Render normalized boxes onto resized copies with Pillow. Run from any directory."""
import json
from pathlib import Path

from PIL import Image, ImageDraw, ImageOps

OUTPUT = Path(__file__).resolve().parent
DATASET = OUTPUT.parent / "butterfly_dataset" / "main"

def read_jsonl(path):
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]

def render():
    files = {row["photo_id"]: row["file_name"] for row in read_jsonl(DATASET / "metadata.jsonl")}
    target = OUTPUT / "previews"
    target.mkdir(exist_ok=True)
    indexed = OUTPUT / "thumbnails"
    indexed.mkdir(exist_ok=True)
    for row in read_jsonl(OUTPUT / "annotations.jsonl"):
        photo_id = row["photo_id"]
        with Image.open(DATASET / files[photo_id]) as original:
            image = ImageOps.exif_transpose(original).convert("RGB")
        image.thumbnail((1120, 1120), Image.Resampling.LANCZOS)
        draw = ImageDraw.Draw(image)
        stroke = max(2, min(4, min(image.size) // 150))
        for i, box in enumerate(row["boxes"] or [], 1):
            left, top = box["left"], box["top"]
            coords = (
                round(left * (image.width - 1)), round(top * (image.height - 1)),
                round((left + box["width"]) * (image.width - 1)),
                round((top + box["height"]) * (image.height - 1)),
            )
            draw.rectangle(coords, outline="#261b12", width=stroke + 2)
            draw.rectangle(coords, outline="#ffad45", width=stroke)
            x, y = coords[:2]
            label = str(i)
            size = 12 + len(label) * 6
            x = min(x, image.width - size)
            y = max(0, y - 15)
            draw.rectangle((x, y, x + size - 1, y + 14), fill="#ffad45")
            draw.text((x + 5, y + 1), label, fill="#19140e")
        image.save(target / f"{photo_id}.jpg", "JPEG", quality=89)
        image.thumbnail((245, 180), Image.Resampling.LANCZOS)
        image.save(indexed / f"{photo_id}.jpg", "JPEG", quality=84)

if __name__ == "__main__":
    render()
