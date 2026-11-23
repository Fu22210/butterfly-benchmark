from pathlib import Path
import sys,json
from PIL import Image,ImageOps,ImageDraw,ImageFont
root=Path(__file__).resolve().parents[1];run=root/'owl_qwen_editor'
journal={r['photo_id']:r for r in map(json.loads,(run/'results/journal.jsonl').open()) if r['error'] is None}
gold={r['photo_id']:r for r in map(json.loads,(root/'gold.jsonl').open())}
files={int(p.stem):p for p in (root/'../butterfly_dataset/main/images').glob('*') if p.stem.isdecimal()}
font=ImageFont.load_default(size=24)
out=run/'examples';out.mkdir(exist_ok=True)
for item in sys.argv[1:]:
 i=int(item);row=journal[i]
 with Image.open(files[i]) as src:
  im=ImageOps.exif_transpose(src).convert('RGB')
  thumb=ImageOps.contain(im,(440,350),Image.Resampling.LANCZOS)
 w,h=thumb.size
 sheet=Image.new('RGB',(w*3+12,h+44),'#f9f9f7');ds=ImageDraw.Draw(sheet)
 for j,(name,boxes,color) in enumerate([
     ('Эталон',gold[i]['boxes'],'#66f097'),
     ('OWLv2',[r['original'] for r in row['reviews']],'#ffc857'),
     ('После Qwen',row['boxes'],'#1cdcef')]):
  pic=thumb.copy();d=ImageDraw.Draw(pic)
  for b in boxes:
   x,y=b['left'],b['top'];right,bottom=x+b['width'],y+b['height']
   xy=[round(x*(w-1)),round(y*(h-1)),round(right*(w-1)),round(bottom*(h-1))]
   d.rectangle(xy,outline='#101010',width=4);d.rectangle(xy,outline=color,width=2)
  x=j*(w+6);sheet.paste(pic,(x,43));ds.text((x+7,7),f'{name} ({len(boxes)})',fill='#222222',font=font)
 sheet.save(out/f'{i}.jpg',quality=93,optimize=True)
 print(i,sheet.size)
