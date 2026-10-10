"""Render every page and audit embedded fonts, text bounds and replacement glyphs."""
from pathlib import Path
import json,fitz
from PIL import Image,ImageDraw
ROOT=Path(__file__).resolve().parent
out=ROOT/'outputs/qa';out.mkdir(parents=True,exist_ok=True);info={}
for name in ['lecture','lab','answers']:
 doc=fitz.open(ROOT/(name+'.pdf'));pages=[];thumbs=[]
 for i,p in enumerate(doc):
  text=p.get_text();fonts=p.get_fonts();bad=[]
  for b in p.get_text('dict')['blocks']:
   if 'lines' not in b:continue
   for line in b['lines']:
    for span in line['spans']:
     x0,y0,x1,y1=span['bbox']
     if x0<0 or y0<0 or x1>p.rect.width+.1 or y1>p.rect.height+.1:bad.append(span['text'])
  pages.append({'page':i+1,'text_chars':len(text),'replacement_glyphs':text.count('\ufffd'),'outside_page_text':bad,'embedded_fonts':all(bool(doc.extract_font(f[0])[3]) for f in fonts)})
  file=out/f'{name}-{i+1:02}.png';p.get_pixmap(matrix=fitz.Matrix(1.3,1.3)).save(file)
  im=Image.open(file).convert('RGB');im.thumbnail((298,430));tile=Image.new('RGB',(310,456),'#dbe2e8');tile.paste(im,((310-im.width)//2,22));ImageDraw.Draw(tile).text((8,5),f'{name} {i+1}',fill='black');thumbs.append(tile)
 for batch in range(0,len(thumbs),6):
  group=thumbs[batch:batch+6];sheet=Image.new('RGB',(930,912),'white')
  for j,t in enumerate(group):sheet.paste(t,((j%3)*310,(j//3)*456))
  sheet.save(out/f'{name}-contact-{batch//6+1}.png')
 info[name]={'pages':len(doc),'page_checks':pages,'fonts':sorted(set(f[3] for p in doc for f in p.get_fonts()))}
(ROOT/'outputs/pdf-audit.json').write_text(json.dumps(info,ensure_ascii=False,indent=2));print({k:v['pages'] for k,v in info.items()})
