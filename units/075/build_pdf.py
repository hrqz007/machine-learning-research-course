"""Build trusted course Markdown with local MathJax SVG and WeasyPrint, no network."""
from pathlib import Path
import argparse,hashlib,html,json,os,re,shutil,subprocess,tempfile
ROOT=Path(__file__).resolve().parent

def render(source,out,cache):
    import markdown
    from weasyprint import HTML,CSS
    text=source.read_text();items=[]
    def collect(match,display=False):
        expr=match.group(1).strip();key=hashlib.sha256((str(display)+expr).encode()).hexdigest()
        item={'expression':expr,'display':display,'path':cache/(key+'.svg'),'token':'MATHPLACEHOLDER'+str(len(items))+'END'};items.append(item)
        return '\n\n'+item['token']+'\n\n' if display else item['token']
    chunks=re.split(r'(```.*?```)',text,flags=re.S)
    for i in range(0,len(chunks),2):
        chunks[i]=re.sub(r'\$\$(.+?)\$\$',lambda m:collect(m,True),chunks[i],flags=re.S)
        chunks[i]=re.sub(r'(?<!\\)\$([^\n$]+?)(?<!\\)\$',collect,chunks[i])
    missing=[v for v in items if not v['path'].exists()]
    if missing:
        result=subprocess.run(['node',str(ROOT/'mathjax_render.cjs')],input=json.dumps([{'expression':v['expression'],'display':v['display']} for v in missing]),text=True,capture_output=True,check=True)
        svgs=json.loads(result.stdout)
        if len(svgs)!=len(missing):raise RuntimeError('formula count mismatch')
        for item,svg in zip(missing,svgs):item['path'].write_text(svg)
    body=markdown.markdown(''.join(chunks),extensions=['tables','fenced_code','sane_lists'])
    for item in items:
        svg=item['path'].read_text();align=re.search(r'vertical-align:\s*([^;" ]+)',svg);vertical=align.group(1) if align else '-0.25em'
        cls='math-display-img' if item['display'] else 'math-inline'
        tag=f'<img src="{item["path"].as_uri()}" alt="{html.escape(item["expression"],quote=True)}" class="{cls}" style="vertical-align:{vertical}"/>'
        if item['display']:tag='<div class="math-display">'+tag+'</div>'
        body=body.replace('<p>'+item['token']+'</p>',tag).replace(item['token'],tag)
    body=re.sub(r'<p><img alt="([^"]*)" src="([^"]*)"\s*/></p>',lambda m:f'<figure><img src="{m[2]}" alt="{m[1]}"><figcaption>{m[1]}</figcaption></figure>',body)
    document='<!doctype html><html lang="zh-CN"><meta charset="utf-8"><body class="'+source.stem+'">'+body+'</body></html>'
    HTML(string=document,base_url=str(ROOT)).write_pdf(str(out),stylesheets=[CSS(filename=str(ROOT/'pdf.css'))],pdf_variant='pdf/a-3u')

def main():
    p=argparse.ArgumentParser();p.add_argument('--directory',default=str(ROOT/'outputs/pdf'));a=p.parse_args()
    directory=Path(a.directory).absolute()
    for part in (directory,*directory.parents):
        if part.is_symlink():raise ValueError('symlink output component')
    if directory.resolve()==ROOT:raise ValueError('use outputs/pdf or a new directory')
    if not shutil.which('node'):raise RuntimeError('Node.js and MathJax required; see build-requirements.txt')
    directory.mkdir(parents=True,exist_ok=True)
    with tempfile.TemporaryDirectory(prefix='ml075-pdf-',dir=directory) as td:
        tmp=Path(td);cache=tmp/'math';cache.mkdir()
        for name in ['lecture','lab','answers']:
            output=directory/(name+'.pdf')
            if output.is_symlink() or (output.exists() and (not output.is_file() or output.stat().st_nlink>1)):raise ValueError('unsafe output')
            fresh=tmp/(name+'.pdf');render(ROOT/(name+'.md'),fresh,cache);os.replace(fresh,output);print(output)
if __name__=='__main__':main()
