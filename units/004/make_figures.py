#!/usr/bin/env python3
"""Rebuild the original diagrams for Unit 004 (no network or external artwork).

Requires matplotlib and a Chinese-capable font.
Font portability: this environment has Noto Sans CJK at
/usr/share/fonts/opentype/noto/NotoSansCJK-{Regular,Bold}.ttc. On another machine,
set CJK_FONT to a CJK-capable .ttf/.otf/.ttc file; otherwise a discovered Noto CJK
font is used. The script fails rather than silently producing missing glyphs.
Run: PYTHONPATH=../../.deps python make_figures.py   (from this directory),
or run with matplotlib available in your usual Python environment.
Each PNG is 1440 px wide, at 150 dpi, with a white background.
"""
from pathlib import Path
import os
import tempfile
# Keep rendering caches writable in restricted or read-only home environments.
os.environ.setdefault("MPLCONFIGDIR", str(Path(tempfile.gettempdir()) / "ml-course-matplotlib"))
os.environ.setdefault("XDG_CACHE_HOME", str(Path(tempfile.gettempdir()) / "ml-course-font-cache"))
try:
    (Path(os.environ["XDG_CACHE_HOME"]) / "fontconfig").mkdir(parents=True, exist_ok=True)
except OSError:
    pass  # A pre-existing user cache setting may be read-only; rendering still works.
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch, Rectangle
from matplotlib.font_manager import FontProperties, findSystemFonts

ROOT = Path(__file__).resolve().parent
OUT = ROOT / 'figures'
OUT.mkdir(exist_ok=True)
preferred = Path(os.environ.get('CJK_FONT', '/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc'))
if not preferred.is_file():
    found = [Path(p) for p in findSystemFonts() if 'NotoSansCJK' in p or 'NotoSansSC' in p]
    if not found:
        raise RuntimeError('A Chinese-capable font is required. Set CJK_FONT to its file path.')
    preferred = found[0]
bold_path = preferred.with_name(preferred.name.replace('Regular', 'Bold'))
REG = FontProperties(fname=str(preferred))
BOLD = FontProperties(fname=str(bold_path if bold_path.exists() else preferred))
plt.rcParams.update({'axes.unicode_minus': False, 'savefig.facecolor': 'white'})

INK = '#16324B'; MUTED = '#52687A'; BLUE = '#2563A6'; CYAN = '#0B7C96'
GREEN = '#197757'; RED = '#B44240'; ORANGE = '#B36A18'; LIGHT = '#EEF4F9'
BORDER = '#B8C9D8'; PALE_BLUE = '#EAF2FB'; PALE_GREEN = '#EAF6F0'
PALE_RED = '#FCEEEE'; PALE_ORANGE = '#FFF4E4'; PALE_PURPLE = '#F1EEFA'


def canvas(title, subtitle=None, h=6.2):
    fig, ax = plt.subplots(figsize=(9.6, h), dpi=150)
    fig.subplots_adjust(left=0, right=1, bottom=0, top=1)
    ax.set_xlim(0,100); ax.set_ylim(0,100); ax.axis('off')
    text(ax, 4, 95, title, 21, weight='bold', va='top')
    if subtitle: text(ax, 4, 87, subtitle, 13.4, color=MUTED, va='top')
    return fig, ax


def text(ax, x, y, s, fs=15, color=INK, ha='left', va='center', weight='normal', **kw):
    return ax.text(x,y,s,fontsize=fs,color=color,ha=ha,va=va,
                   fontproperties=BOLD if weight=='bold' else REG,
                   linespacing=1.35, zorder=5, **kw)


def box(ax, x,y,w,h, label=None, fill=LIGHT, edge=BORDER, fs=15, color=INK,
        lw=1.3, radius=1.3, weight='normal', align='center'):
    p = FancyBboxPatch((x,y),w,h,boxstyle=f'round,pad=0.01,rounding_size={radius}',
                       facecolor=fill,edgecolor=edge,linewidth=lw,zorder=2)
    ax.add_patch(p)
    if label:
        text(ax, x+w/2 if align=='center' else x+2, y+h/2, label, fs, color,
             ha='center' if align=='center' else 'left', weight=weight)
    return p


def arrow(ax, x1,y1,x2,y2,color=BLUE,lw=1.8,style='-|>',rad=0,mut=13):
    p=FancyArrowPatch((x1,y1),(x2,y2),arrowstyle=style,mutation_scale=mut,
                      linewidth=lw,color=color,connectionstyle=f'arc3,rad={rad}',zorder=3)
    ax.add_patch(p); return p


def line(ax,xs,ys,color=BORDER,lw=1.3,style='-',zorder=1):
    ax.plot(xs,ys,color=color,lw=lw,linestyle=style,zorder=zorder)


def note(ax, s, y=5.3, color=MUTED, fs=13.2):
    text(ax,4,y,s,fs,color,va='bottom')


def save(fig,name):
    # Catch text that escapes the actual image boundary before exporting.
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    bounds = fig.bbox
    for ax in fig.axes:
        for label in ax.texts:
            b = label.get_window_extent(renderer)
            if b.x0 < 0 or b.y0 < 0 or b.x1 > bounds.width or b.y1 > bounds.height:
                raise ValueError(f'Text outside figure in {name}: {label.get_text()}')
    fig.savefig(OUT/name, dpi=150, facecolor='white')
    plt.close(fig)


# 01 — Duplicated formulas can drift; one implementation reduces that risk.
fig,ax=canvas('同一公式，尽量只保留一个实现', '示例输入均为 area=70、weight=2、bias=10。')
line(ax,[50,50],[17,80],color=BORDER,style='--')
text(ax,25,77,'复制后，各自修改',17,ha='center',weight='bold')
text(ax,75,77,'统一调用一个函数',17,ha='center',weight='bold')
box(ax,4,53,39,17,fill=PALE_BLUE,edge=BLUE)
text(ax,6,66,'训练中的公式',13.5,color=BLUE)
text(ax,23.5,59,'weight * area + bias',15.5,ha='center')
arrow(ax,23.5,53,13,47)
box(ax,4,37,18,10,'150',PALE_GREEN,GREEN,19,weight='bold')
box(ax,4,15,39,17,fill=PALE_RED,edge=RED)
text(ax,6,28,'测试中的公式：误改了符号',13.2,color=RED)
text(ax,23.5,21,'weight * area − bias',15.5,ha='center')
box(ax,30,37,17,10,'130',PALE_RED,RED,19,weight='bold')
arrow(ax,23.5,32,38.5,37,color=RED)
text(ax,26,42,'≠',20,color=RED,ha='center')
box(ax,57,61,15,10,'训练',PALE_BLUE,BLUE,16)
box(ax,80,61,15,10,'测试',PALE_BLUE,BLUE,16)
arrow(ax,64.5,61,71,51); arrow(ax,87.5,61,81,51)
box(ax,55,31,42,20,fill=PALE_BLUE,edge=BLUE)
text(ax,76,46,'predict_one(area, weight, bias)',13.5,ha='center',color=BLUE,weight='bold')
text(ax,76,37,'return weight * area + bias',14.2,ha='center')
arrow(ax,76,31,76,25)
box(ax,61,15,30,10,'两处都得到 150',PALE_GREEN,GREEN,16,weight='bold')
note(ax,'共享实现减少分叉；公式本身是否正确，仍要用已知答案检查。')
save(fig,'01_single_source.png')

# 02 — A value returned to the caller and text printed to a screen take different paths.
fig,ax=canvas('return 和 print：结果走向不同', '屏幕出现 150，不代表变量拿到了数值 150。')
text(ax,25,76,'返回数值',17,ha='center',weight='bold',color=GREEN)
text(ax,75,76,'只打印，没有显式返回',17,ha='center',weight='bold',color=ORANGE)
line(ax,[50,50],[14,80],style='--')
box(ax,4,57,42,13,'return 2 * 70 + 10',PALE_GREEN,GREEN,17)
arrow(ax,25,57,25,49,color=GREEN)
text(ax,27,53,'返回 150',12.5,color=GREEN)
box(ax,8,36,34,13,'p = 150',PALE_GREEN,GREEN,19,weight='bold')
arrow(ax,25,36,25,28,color=GREEN)
box(ax,4,15,42,13,'abs(p − 144) = 6',PALE_GREEN,GREEN,17)
box(ax,54,57,42,13,'print(2 * 70 + 10)',PALE_ORANGE,ORANGE,17)
arrow(ax,67,57,64,49,color=ORANGE)
arrow(ax,84,57,85,49,color=MUTED)
box(ax,54,36,21,13,'屏幕：150',PALE_ORANGE,ORANGE,16)
box(ax,79,36,17,13,'q = None',LIGHT,MUTED,15)
text(ax,54,31,'输出文字',12.5,color=ORANGE)
text(ax,80,31,'隐式返回',12.5,color=MUTED)
arrow(ax,87.5,36,87.5,25,color=RED)
box(ax,54,12,42,13,'q − 144 → TypeError',PALE_RED,RED,16)
note(ax,'数值核心先 return；外层再决定打印、保存或画图。',y=3)
save(fig,'02_return.png')

# 03 — Distinct local and outer bindings; explicit call and return.
fig,ax=canvas('名字相同，也可以属于不同作用域', '本次调用：estimate = predict_one(70, 2, 10)')
box(ax,4,19,39,58,fill=LIGHT,edge=BLUE)
box(ax,57,19,39,58,fill=PALE_BLUE,edge=BLUE)
text(ax,7,71,'外层作用域',17,weight='bold',color=BLUE)
text(ax,60,71,'本次调用的局部作用域',16,weight='bold',color=BLUE)
box(ax,8,52,31,12,'area → 999',fill='white',edge=BORDER,fs=18)
box(ax,8,26,31,12,'estimate → 150',fill=PALE_GREEN,edge=GREEN,fs=17,weight='bold')
text(ax,23.5,45,'外层 area 不变',13.5,color=MUTED,ha='center')
text(ax,61,60,'area → 70',16)
text(ax,61,51,'weight → 2',16)
text(ax,61,42,'bias → 10',16)
line(ax,[60,93],[36,36])
text(ax,61,29,'prediction → 150',16,weight='bold')
arrow(ax,43,61,57,61)
text(ax,50,66,'传入',13,ha='center',color=BLUE)
text(ax,50,56,'70, 2, 10',12.4,ha='center',color=BLUE)
arrow(ax,57,30,43,30,color=GREEN)
text(ax,50,36,'return',12.8,ha='center',color=GREEN)
text(ax,50,23,'150',14,ha='center',color=GREEN,weight='bold')
note(ax,'不要把同名变量当成全局共用的一只盒子；显式参数让依赖看得见。')
save(fig,'03_scope.png')

# 04 — Input validation, computation, and validation of the computed output.
fig,ax=canvas('函数契约覆盖成功与失败两条路径', 'predict_one：面积 m² × 单价 万元/m² + 截距 万元 → 预测 万元',h=6.4)
text(ax,4,77,'成功路径',15.5,color=GREEN,weight='bold')
xs=[4,28,52,76]; ws=[20,20,20,20]
labels=['70, 2, 10\n明确数值与单位','检查输入\n类型 + 有限性','计算\n2 × 70 + 10','检查输出\n有限值 150']
for x,w,l in zip(xs,ws,labels): box(ax,x,54,w,18,l,PALE_GREEN,GREEN,14.5)
for a,b in [(24,28),(48,52),(72,76)]: arrow(ax,a,63,b,63,color=GREEN)
text(ax,14,47,'area, weight, bias',12,ha='center',color=MUTED)
text(ax,38,47,'int / float；排除 bool',12.2,ha='center',color=MUTED)
text(ax,62,47,'不修改输入',13,ha='center',color=MUTED)
text(ax,86,47,'返回数值，单位万元',12.5,ha='center',color=MUTED)
text(ax,4,37,'失败路径',15.5,color=RED,weight='bold')
box(ax,4,14,44,17,fill=PALE_RED,edge=RED)
text(ax,6,26,'"70"、True、NaN、∞',16,color=RED)
text(ax,6,18,'输入检查拒绝：类型或取值不合约定',13.3)
box(ax,52,14,44,17,fill=PALE_ORANGE,edge=ORANGE)
text(ax,54,26,'有限输入也可能算出非有限结果',14.5,color=ORANGE)
text(ax,54,18,'如 1e308 × 1e308：输出检查拒绝',13.3)
arrow(ax,38,43,38,33,color=RED)
arrow(ax,86,43,86,33,color=ORANGE)
note(ax,'失败要明确抛出异常；不能把无效结果当作可用预测继续传递。',y=4)
save(fig,'04_contract.png')

# 05 — A tiny MAE computation exposes sample pairing and the reduction.
fig,ax=canvas('MAE：按对应位置做差，再求平均', '先检查：非空、等长、每个值有限。这里两个位置分别对应同一条样本。')
box(ax,4,67,92,11,'输入通过检查 → 才进入误差计算',PALE_GREEN,GREEN,16,weight='bold')
text(ax,34,61,'位置 0',13.5,ha='center',color=MUTED)
text(ax,53,61,'位置 1',13.5,ha='center',color=MUTED)
text(ax,4,52,'预测',16,weight='bold')
text(ax,4,38,'标签',16,weight='bold')
text(ax,4,21,'绝对误差',15.5,weight='bold')
for x,p,y,d in [(25,140,144,4),(44,180,174,6)]:
    box(ax,x,46,18,12,str(p),PALE_BLUE,BLUE,20)
    box(ax,x,32,18,12,str(y),PALE_BLUE,BLUE,20)
    arrow(ax,x+9,32,x+9,27,color=CYAN)
    box(ax,x,15,18,12,str(d),PALE_GREEN,GREEN,20,weight='bold')
# Row labels plus the explicit error values identify the absolute differences.
arrow(ax,62,21,72,21,color=GREEN)
box(ax,72,15,24,41,fill=PALE_GREEN,edge=GREEN)
text(ax,84,47,'4 + 6 = 10',16,ha='center')
text(ax,84,36,'10 ÷ 2',18,ha='center')
text(ax,84,23,'MAE = 5',20,ha='center',weight='bold',color=GREEN)
text(ax,84,10,'单位：万元',13,ha='center',color=MUTED)
note(ax,'MAE 不需要面积、模型参数或全局训练状态。',y=3)
save(fig,'05_mae.png')

# 06 — Named records retain row semantics, while validation is still necessary.
fig,ax=canvas('把同一样本的字段放在一条记录里', '示例取训练集前两条：H1 (50, 110)、H2 (60, 130)。')
text(ax,23,77,'平行列表：依赖位置配对',16,ha='center',weight='bold')
text(ax,74,77,'记录列表：字段名表达含义',16,ha='center',weight='bold')
for y,lab,v1,v2 in [(64,'id','H1','H2'),(50,'area_m2','50','60'),(36,'price_wan','110','130')]:
    text(ax,4,y,lab,14.2)
    box(ax,21,y-5,10,10,v1,PALE_BLUE,BLUE,15)
    box(ax,33,y-5,10,10,v2,PALE_BLUE,BLUE,15)
line(ax,[26,26],[30,70],BLUE,1.1,':');line(ax,[38,38],[30,70],CYAN,1.1,':')
text(ax,26,25,'位置 0',12,ha='center',color=BLUE)
text(ax,38,25,'位置 1',12,ha='center',color=CYAN)
arrow(ax,44,51,52,51)
box(ax,54,51,42,20,fill=PALE_BLUE,edge=BLUE)
text(ax,57,65,'{"id": "H1",',15.4)
text(ax,57,58,' "area_m2": 50, "price_wan": 110}',13.2)
box(ax,54,25,42,20,fill=PALE_GREEN,edge=GREEN)
text(ax,57,39,'{"id": "H2",',15.4)
text(ax,57,32,' "area_m2": 60, "price_wan": 130}',13.2)
text(ax,4,17,'只重排其中一列，配对就会破坏。',13.2,color=RED)
text(ax,54,17,'整条记录移动，字段仍跟着样本。',13.2,color=GREEN)
box(ax,4,3,92,9,'仍要检查：键名、单位、id 唯一性；字典不会自动修复数据。',PALE_ORANGE,ORANGE,14)
save(fig,'06_records.png')

# 07 — One exception is propagated through multiple stack frames.
fig,ax=canvas('一次错误，可以经过多层调用向外传播', '示意：预测 2 条、标签 3 条，违反了 MAE 的等长约定。')
text(ax,24,77,'调用链',16,ha='center',weight='bold')
text(ax,76,77,'从报错中恢复定位线索',16,ha='center',weight='bold')
for y,lab in [(61,'run_experiment(...)'),(39,'evaluate(...)'),(17,'mean_absolute_error(...)')]:
    box(ax,8,y,39,12,lab,PALE_BLUE,BLUE,14.6)
arrow(ax,25,61,25,51);text(ax,28,56,'调用',12.5,color=BLUE)
arrow(ax,25,39,25,29);text(ax,28,34,'调用',12.5,color=BLUE)
# Red outward propagation travels separately from blue calls.
line(ax,[8,4,4,8],[23,23,67,67],RED,1.8)
arrow(ax,4,40,8,45,color=RED)
arrow(ax,4,63,8,67,color=RED)
text(ax,4,12,'同一个异常向外传播',13.2,color=RED)
box(ax,55,49,41,24,fill=LIGHT,edge=BORDER)
text(ax,58,66,'先看最后的异常类型与原因',14.1,weight='bold')
text(ax,58,57,'ValueError:',15,color=RED,weight='bold')
text(ax,58,51,'预测与标签长度必须相同',13.5,color=RED)
box(ax,55,17,41,25,fill=PALE_ORANGE,edge=ORANGE)
text(ax,58,35,'再沿调用链检查',14.4,weight='bold')
text(ax,58,26,'哪一步少了或多了样本？',14)
text(ax,58,19,'文件、行号和调用者提供上下文。',12.7)
arrow(ax,75.5,49,75.5,42,color=ORANGE)
arrow(ax,47,23,55,54,color=RED)
note(ax,'多层堆栈不等于多次错误。不要用 except Exception: pass 隐藏原因。',y=3.1,fs=12.7)
save(fig,'07_traceback.png')

# 08 — Relative paths are rooted at the process cwd; script defaults use __file__.
fig,ax=canvas('工作目录与脚本目录，是两个位置', '从课程根目录运行：python units/004/experiment.py',h=6.4)
box(ax,4,21,36,57,fill=LIGHT,edge=BORDER)
text(ax,7,72,'course/   ← 当前工作目录',14.5,weight='bold')
line(ax,[9,9,13],[65,58,58],BORDER,1.6,zorder=3)
text(ax,14,58,'units/',15)
line(ax,[16,16,20],[54,46,46],BORDER,1.6,zorder=3)
text(ax,21,46,'004/',15)
line(ax,[23,23],[41,27],BORDER,1.6,zorder=3)
line(ax,[23,25],[36,36],BORDER,1.6,zorder=3)
line(ax,[23,25],[27,27],BORDER,1.6,zorder=3)
text(ax,26,36,'experiment.py',11.7,color=BLUE)
text(ax,26,27,'data/train.csv',11.7,color=GREEN)
text(ax,7,15,'目标数据随脚本一起提供',13.2,color=MUTED)
box(ax,46,52,50,26,fill=PALE_RED,edge=RED)
text(ax,49,70,'直接用相对路径',15,weight='bold',color=RED)
text(ax,49,61,'open("data/train.csv")',15)
text(ax,49,54,'→ course/data/train.csv',13.2,color=RED)
box(ax,46,16,50,28,fill=PALE_GREEN,edge=GREEN)
text(ax,49,37,'从脚本自身定位',15,weight='bold',color=GREEN)
text(ax,49,29,'Path(__file__).resolve().parent',12.9)
text(ax,49,21,'/ "data" / "train.csv"',14.2)
arrow(ax,46,25,40,27,color=GREEN)
text(ax,46,10,'→ course/units/004/data/train.csv',12.6,color=GREEN)
note(ax,'Notebook 通常没有 __file__：本讲由导入的 experiment 模块确定路径。',y=3,fs=12.8)
save(fig,'08_paths.png')

# 09 — Parsing a CSV does not perform domain validation or numerical conversion.
fig,ax=canvas('CSV 读入成功，只是验证的起点', '字段检查、类型转换与业务检查，应当有清楚的先后顺序。',h=6.6)
box(ax,4,60,35,20,fill=LIGHT,edge=BORDER)
text(ax,6,75,'CSV 文件',14,color=MUTED,weight='bold')
text(ax,6,68,'id,area_m2,price_wan',13.6)
text(ax,6,64,'H1,50,110',14.6)
arrow(ax,39,70,55,70)
text(ax,47,75,'DictReader',12.6,ha='center',color=BLUE)
box(ax,55,60,41,20,fill=PALE_BLUE,edge=BLUE)
text(ax,58,75,'字段值先是文本',14.2,color=BLUE,weight='bold')
text(ax,58,69,'{"id": "H1", "area_m2": "50",',12.6)
text(ax,58,64,' "price_wan": "110"}',13.3)
# validation stages across a single lane
stages=[('字段','完全匹配'),('float','显式转换'),('数值','必须有限'),('面积','必须 > 0'),('id','非空且唯一')]
for i,(h,s) in enumerate(stages):
    x=4+i*19
    box(ax,x,37,16,14,fill=PALE_GREEN,edge=GREEN)
    text(ax,x+8,46,h,15.3,ha='center',weight='bold',color=GREEN)
    text(ax,x+8,40,s,12.4,ha='center')
    if i<4: arrow(ax,x+16,44,x+19,44,color=GREEN,mut=11)
line(ax,[96,98,98,2,2],[70,70,55,55,44],BLUE,1.6)
arrow(ax,2,44,4,44)
line(ax,[88,88,50],[37,31,31],GREEN,1.6)
arrow(ax,50,31,50,25,color=GREEN)
box(ax,4,5,55,21,fill=PALE_GREEN,edge=GREEN)
text(ax,7,21,'合格数值记录',14.2,weight='bold',color=GREEN)
text(ax,7,15,'{"id": "H1", "area_m2": 50.0,',13.2)
text(ax,7,9,' "price_wan": 110.0}',13.2)
box(ax,65,5,31,21,fill=PALE_RED,edge=RED)
text(ax,67,21,'任一步失败 → 明确报错',12.9,weight='bold',color=RED)
text(ax,67,15,'标出文件、行或 id、字段',12.7)
text(ax,67,9,'不偷偷填零、不无声丢行',12.7)
arrow(ax,88,37,88,25,color=RED)
save(fig,'09_csv.png')

# 10 — Structured numeric output is separate from both data sources and presentation.
fig,ax=canvas('输入、计算、保存与展示各有位置', 'JSON 保留字段名和数值类型，方便复核，也方便其他程序读取。',h=6.6)
text(ax,16,78,'原始数据',15.5,ha='center',weight='bold')
for y,l in [(61,'train.csv'),(44,'test_inputs.csv'),(27,'test_labels.csv')]:
    box(ax,4,y,25,12,l,LIGHT,BORDER,14.2)
arrow(ax,29,67,38,58);arrow(ax,29,50,38,50);arrow(ax,29,33,38,42)
box(ax,38,31,24,34,fill=PALE_BLUE,edge=BLUE)
text(ax,50,58,'明确参数与输入',13.8,ha='center',weight='bold',color=BLUE)
text(ax,50,48,'预测 → 误差',16,ha='center')
text(ax,50,37,'返回结果字典',14.3,ha='center',weight='bold')
arrow(ax,62,56,70,64,color=GREEN)
box(ax,70,41,27,38,fill=PALE_GREEN,edge=GREEN)
text(ax,72,74,'result.json 节选',14.5,color=GREEN,weight='bold')
text(ax,72,65,'{',13.5)
text(ax,72,58,' "test_count": 2,',12.8)
text(ax,72,51,' "test_mae_wan": 5',12.2)
text(ax,72,44,'}',13.5)
text(ax,84,35,'5 是数值，不是 "5万元"',12.3,ha='center',color=GREEN)
arrow(ax,62,39,70,24,color=ORANGE)
box(ax,70,15,27,15,'print / 图表\n用于即时查看',PALE_ORANGE,ORANGE,14.2)
box(ax,4,3,58,15,fill=PALE_PURPLE,edge='#8D79B7')
text(ax,7,13,'共同解释一次实验',13.8,weight='bold')
text(ax,7,6,'输入版本 + 代码版本 + 参数配置',14.1)
line(ax,[62,65,65,70],[11,11,46,46],color='#8D79B7',style='--')
save(fig,'10_outputs.png')

# 11 — Increasingly broad checks, each with a distinct expected outcome.
fig,ax=canvas('测试范围逐层扩大，每层提供不同证据', '小输入先验证计算，再检查边界、数据读写和完整的新进程运行。',h=6.7)
rows=[
    (63,'1  数值核心','predict_one(70, 2, 10)\nMAE([140,180], [144,174])','150\n5',PALE_BLUE,BLUE),
    (46,'2  失败边界','空序列 / 不等长 / NaN\n错误类型与缺少字段','指定异常\n按约定拒绝',PALE_RED,RED),
    (29,'3  读写边界','CSV 文本 → 验证 → 数值\n结果字典 → JSON → 读取','字段与单位正确\n数字仍是数字',PALE_ORANGE,ORANGE),
    (12,'4  完整实验','新进程 + 不同工作目录\n真实 CSV → 全流程 → JSON','选中 B\n测试 MAE = 5',PALE_GREEN,GREEN),
]
for y,lab,inp,out,fill,col in rows:
    box(ax,4,y,92,14,fill=fill,edge=col)
    text(ax,6,y+7,lab,14.5,color=col,weight='bold')
    line(ax,[27,27],[y+2,y+12],col,1,zorder=3)
    text(ax,30,y+7,inp,13.2)
    arrow(ax,66,y+7,73,y+7,color=col)
    text(ax,76,y+7,out,14.1,color=col,weight='bold')
note(ax,'测试用 assert 核验；关键输入检查用 if + raise（不能依赖可能被 -O 移除的断言）。',y=3,fs=12.4)
save(fig,'11_tests.png')

print('Generated 11 teaching PNGs in figures/')
