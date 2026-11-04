"""Render the added-layer banner, mechanism diagram and measured benchmark.

Original vector artwork. The benchmark uses an explicitly named public data
file, never model calls. Earlier composite diagrams and animation are retained.
"""
from pathlib import Path
import argparse
import csv
import hashlib
import json
import math
import os

ROOT = Path(__file__).resolve().parents[1]
os.environ.setdefault('MPLCONFIGDIR', str(ROOT / 'review/cache/matplotlib'))
os.environ.setdefault('XDG_CACHE_HOME', str(ROOT / 'review/cache'))
Path(os.environ['XDG_CACHE_HOME']).mkdir(parents=True, exist_ok=True)
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch, Circle

BG, PANEL, INK, MUTED, EDGE = '#0b1219', '#14212c', '#edf3f5', '#afbdc6', '#2b3e4c'
MINT, BLUE, GREY = '#63e5d3', '#78b8ee', '#758092'
plt.rcParams.update({'font.family': 'DejaVu Sans', 'font.size': 11,
                     'svg.fonttype': 'path', 'pdf.fonttype': 42})


def canvas(width, height, background=BG):
    fig = plt.figure(figsize=(width/100, height/100), facecolor=background)
    ax = fig.add_axes([0, 0, 1, 1], xlim=(0, width), ylim=(height, 0))
    ax.set_axis_off()
    return fig, ax


def text(ax, x, y, value, size=15, color=INK, weight='normal', **kw):
    return ax.text(x, y, value, fontsize=size, color=color, weight=weight,
                   va='center', **kw)


def box(ax, x, y, w, h, fill=PANEL, edge=EDGE, radius=12):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle=f'round,pad=0,rounding_size={radius}',
                               linewidth=1.2, edgecolor=edge, facecolor=fill))


def arrow(ax, a, b, color=MINT, style='-|>', **kw):
    ax.add_patch(FancyArrowPatch(a, b, arrowstyle=style, mutation_scale=13,
                                linewidth=1.7, color=color, **kw))


def save(fig, name, metadata):
    paths=[]
    for ext in ['svg', 'png', 'pdf']:
        p=ROOT/'assets'/f'{name}.{ext}'
        fig.savefig(p, dpi=180, facecolor=fig.get_facecolor(), metadata={'Creator':'QwenQPU'})
        paths.append({'path':str(p.relative_to(ROOT)), 'sha256':hashlib.sha256(p.read_bytes()).hexdigest()})
    plt.close(fig)
    metadata['artifacts'][name]=paths


def banner(meta):
    fig, ax=canvas(1440, 440)
    text(ax, 58, 54, 'Qwen3.5-4B', 14, MUTED)
    text(ax, 51, 159, 'Qwen', 65, weight='bold')
    text(ax, 326, 159, 'QPU', 65, MINT, weight='bold')
    text(ax, 58, 247, 'QPU inside a language model', 23)
    text(ax, 58, 311, 'Quantum circuits → measurement probabilities', 15, MUTED)
    text(ax, 58, 376, 'H   X   T   CNOT', 14, MINT, family='DejaVu Sans Mono')
    box(ax, 806, 53, 576, 334, fill=BG)
    text(ax, 1094, 87, 'STANDARD QWEN LAYERS', 12, MUTED, ha='center')
    box(ax, 860, 117, 468, 96, edge=MINT)
    text(ax, 1094, 147, '4 QPU layers', 24, MINT, 'bold', ha='center')
    text(ax, 1094, 183, 'State memory · gates · probability readout', 13, ha='center')
    arrow(ax, (1094, 214), (1094, 244))
    box(ax, 860, 247, 468, 96, edge=BLUE)
    text(ax, 1094, 281, '32 language layers', 23, BLUE, 'bold', ha='center')
    text(ax, 1094, 315, 'Qwen reads the task and specifies the circuit', 13, ha='center')
    save(fig, 'banner-added-layers', meta)


def architecture(meta):
    fig, ax=canvas(1540, 740)
    text(ax, 48, 49, 'Quantum computation in standard Qwen layers', 25, weight='bold')
    text(ax, 48, 89, 'Circuit instructions enter as tokens; model-generated digits carry the result.', 14, MUTED)
    for x,w in [(46,352),(420,692),(1134,360)]:box(ax,x,128,w,536)
    text(ax, 66, 162, 'a   Task and request', 18, weight='bold')
    text(ax, 68, 225, 'Start in |000000⟩', 17)
    text(ax, 68, 269, 'Apply H3 → T3 → H3', 17, MINT)
    text(ax, 68, 313, 'Find P(q3 = 1)', 17)
    arrow(ax,(222,341),(222,375))
    box(ax, 68, 386, 308, 101, fill=BG, edge=BLUE)
    text(ax, 90, 417, 'Qwen', 18, BLUE, 'bold')
    text(ax, 90, 454, 'Circuit + indexed event', 15)
    text(ax, 68, 544, 'Host parses the request', 14, MUTED)
    text(ax, 68, 576, 'and loads instruction tokens.', 14, MUTED)
    arrow(ax,(377,437),(441,437),BLUE)

    text(ax, 440, 162, 'b   Shared model forward path', 18, weight='bold')
    box(ax, 446, 202, 640, 314, fill=BG, edge=MINT)
    text(ax, 466, 230, '4 fixed-weight QPU layers', 18, MINT, 'bold')
    box(ax, 466, 267, 270, 124)
    text(ax, 484, 296, 'Gated DeltaNet', 16, weight='bold')
    text(ax, 484, 333, 'Amplitude digits', 14)
    text(ax, 484, 364, 'Program + counter', 14)
    box(ax, 783, 267, 282, 124)
    text(ax, 802, 296, 'Fixed SwiGLU weights', 16, weight='bold')
    text(ax, 802, 333, 'H / X / T / CNOT', 14)
    text(ax, 802, 364, 'Born probabilities', 14)
    arrow(ax,(739,326),(779,326))
    arrow(ax,(924,398),(601,398),style='-|>',connectionstyle='arc3,rad=-.28')
    text(ax, 766, 451, 'Numerical token feedback → state writeback', 13, MINT, ha='center')
    text(ax, 766, 487, 'Shared arithmetic at addressed amplitudes', 13, MUTED, ha='center')
    arrow(ax,(766,519),(766,545),BLUE)
    box(ax, 446, 550, 640, 73, fill=BG, edge=BLUE)
    text(ax, 766, 586, '32 Qwen language layers + model output head', 16, BLUE, ha='center')
    arrow(ax,(1090,586),(1151,586),BLUE)

    text(ax, 1154, 162, 'c   Numerical answer', 18, weight='bold')
    text(ax, 1156, 236, 'Model output digits', 17, MINT)
    text(ax, 1156, 275, '↓ decode', 15, MUTED)
    text(ax, 1156, 322, 'P(q3 = 1)', 19)
    text(ax, 1156, 370, '0.1464464664', 22, MINT, 'bold')
    ax.plot([1156,1472],[425,425],color=EDGE,lw=1)
    text(ax, 1156, 466, 'Independent reference', 15, MUTED)
    text(ax, 1156, 503, 'Qiskit: 0.1464466094', 15)
    text(ax, 1156, 548, 'Absolute error: 1.43 × 10⁻⁷', 14)
    text(ax, 48, 704, 'Six-qubit H–T–H calculation · GDN memory and gate arithmetic in the added layers', 13, MUTED)
    save(fig,'architecture-added-layers',meta)


def wilson(k,n):
    z=1.959963984540054;p=k/n;den=1+z*z/n
    mid=(p+z*z/(2*n))/den;half=z*math.sqrt(p*(1-p)/n+z*z/(4*n*n))/den
    return max(0,mid-half),min(1,mid+half)


def benchmark(source,meta):
    data=json.loads(source.read_text())
    rows=data['rows']; assert [r['family'] for r in rows]==['nonclifford','clifford','all']
    assert sum(r['n'] for r in rows[:2])==rows[2]['n']==data['paired_tasks_scored']
    assert data['terminal_conditions_scored']==2*data['paired_tasks_scored']
    complete=data['paired_tasks_scored']==data['scheduled_tasks'] and data['status']=='completed'
    meta['benchmark_source']={'path':str(source.relative_to(ROOT)), 'sha256':hashlib.sha256(source.read_bytes()).hexdigest(),
                              'version':data['version'],'complete':complete,'paired_tasks':data['paired_tasks_scored']}
    fig,ax=plt.subplots(figsize=(14.4,6.8));fig.patch.set_facecolor(BG);ax.set_facecolor(BG)
    labels=['Non-Clifford circuits','Clifford controls','All paired tasks']
    table=[]
    for j,(key,color,label) in enumerate([('qwen_correct',GREY,'Qwen3.5-4B'),('qwenqpu_correct',MINT,'QwenQPU')]):
        x=[i+(j-.5)*.3 for i in range(3)];values=[100*r[key]/r['n'] for r in rows]
        bars=ax.bar(x,values,width=.255,color=color,label=label,zorder=3)
        for bar,r,pct in zip(bars,rows,values):
            lo,hi=wilson(r[key],r['n']);cx=bar.get_x()+bar.get_width()/2
            ax.errorbar(cx,pct,yerr=[[max(0,pct-100*lo)],[max(0,100*hi-pct)]],fmt='none',color=INK,elinewidth=1.1,capsize=4,zorder=4)
            ax.text(cx,107,f'{r[key]}/{r["n"]}',color=color,ha='center',weight='bold',fontsize=13)
            if pct>25:ax.text(cx,pct*.55,f'{pct:.1f}%',color=BG,ha='center',weight='bold',fontsize=12)
            else:ax.text(cx,100*hi+4,f'{pct:.1f}%',color=INK,ha='center',fontsize=11)
            table.append(dict(family=r['family'],model=label,correct=r[key],n=r['n'],percent=pct,wilson_low=100*lo,wilson_high=100*hi))
    ax.set_ylim(0,116);ax.set_yticks([0,25,50,75,100]);ax.set_ylabel('Tasks correct (%)',color=MUTED,fontsize=12)
    ax.set_xticks(range(3),[f'{label}\nn = {r["n"]}' for label,r in zip(labels,rows)])
    ax.tick_params(colors=MUTED,labelsize=12,length=0,pad=10)
    ax.grid(axis='y',color=EDGE,linewidth=.6);ax.set_axisbelow(True)
    for spine in ax.spines.values():spine.set_visible(False)
    fig.text(.065,.94,'QwenQPU vs Qwen on quantum circuit calculation',fontsize=21,weight='bold',color=INK)
    subtitle=f'{data["paired_tasks_scored"]} paired tasks' if complete else f'Interim: {data["paired_tasks_scored"]} of {data["scheduled_tasks"]} paired tasks'
    fig.text(.065,.882,subtitle+' · 3, 4 and 6 qubits · two event probabilities per circuit',fontsize=12,color=MUTED)
    legend=fig.legend(loc='upper right',bbox_to_anchor=(.955,.81),ncol=2,frameon=False,fontsize=12)
    for t in legend.get_texts():t.set_color(INK)
    fig.text(.065,.054,'16,384 total predictions · |error| ≤ 5 × 10⁻⁵ for both probabilities · independent Qiskit reference',fontsize=11,color=MUTED)
    fig.text(.065,.022,'Whiskers: Wilson 95% intervals.  '+data['version'],fontsize=10,color=MUTED)
    fig.subplots_adjust(left=.085,right=.96,top=.775,bottom=.20)
    save(fig,'benchmark-added-layers',meta)
    p=ROOT/'assets/benchmark-added-layers.csv'
    with p.open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(table[0]));w.writeheader();w.writerows(table)


def main():
    p=argparse.ArgumentParser();p.add_argument('--summary',type=Path,default=ROOT/'evidence/added-layers/r5-interim.json');a=p.parse_args()
    meta={'generator':'tools/render_added_layers.py','architecture':'4 QPU + 32 language layers; stock Qwen3_5ForCausalLM forward',
          'design_references':['https://github.com/divagr18/qwenputer/blob/main/docs/benchmarks-arith.png',
                               'https://github.com/divagr18/qwenputer/blob/main/docs/gdn-ram.png'],
          'artifacts':{},'animation_changed':False}
    movie={p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in [ROOT/'assets/demo.gif',ROOT/'assets/demo.mp4']}
    banner(meta);architecture(meta);benchmark(a.summary.resolve(),meta)
    assert movie=={p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in [ROOT/'assets/demo.gif',ROOT/'assets/demo.mp4']}
    meta['preserved_animation_sha256']=movie
    (ROOT/'assets/added-layers-figure-metadata.json').write_text(json.dumps(meta,indent=2)+'\n')
    print(json.dumps(meta['benchmark_source']))


if __name__=='__main__':main()
