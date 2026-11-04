"""Draw a looping calculation diagram from an existing native execution trace.

The numerical table contains decoded, recorded gate-boundary values. The grid
maps logical amplitude addresses, not physical GDN tensor coordinates. The layer
strip marks the role of each group; animation timing is authored, not telemetry.
No inference, stochastic sampling, external assets or benchmark runs are used.
"""
from pathlib import Path
import hashlib
import json
import math

from render_added_layers import canvas, text, BG, PANEL, INK, MUTED, EDGE, MINT, BLUE
import matplotlib.pyplot as plt
from matplotlib import font_manager
import numpy as np
from PIL import Image, ImageDraw, ImageFont

ROOT = Path(__file__).resolve().parents[1]
CORAL = '#f18a78'
TRACE = ROOT / 'evidence/added-layers/banner-phase-trace.json'
EXPECTED = '61b5674076706453ea4e146a7c2d4996488448fdd2fe5b147125666362daa4f9'


def sha(p):
    return hashlib.sha256(p.read_bytes()).hexdigest()


def font(size, mono=False, bold=False):
    prop = font_manager.FontProperties(family='DejaVu Sans Mono' if mono else 'DejaVu Sans',
                                       weight='bold' if bold else 'normal')
    return ImageFont.truetype(font_manager.findfont(prop), size)


def base():
    fig, ax = canvas(1440, 440)
    text(ax, 58, 54, 'Qwen3.5-4B', 14, MUTED)
    text(ax, 51, 159, 'Qwen', 65, weight='bold')
    text(ax, 326, 159, 'QPU', 65, MINT, weight='bold')
    text(ax, 58, 247, 'QPU inside a language model', 23)
    text(ax, 58, 311, 'Quantum circuits → measurement probabilities', 15, MUTED)
    text(ax, 58, 376, 'H   X   T   CNOT', 14, MINT, family='DejaVu Sans Mono')
    fig.canvas.draw()
    im = Image.fromarray(np.asarray(fig.canvas.buffer_rgba()).copy()).convert('RGB')
    plt.close(fig)
    return im


def mix(a, b, amount):
    aa = bytes.fromhex(a.lstrip('#')); bb = bytes.fromhex(b.lstrip('#'))
    return tuple(round(x+(y-x)*amount) for x, y in zip(aa, bb))


def frame(background, data, pc, loaded=3, mode='gate', active=None, flash=False, digits=None):
    im = background.copy(); d = ImageDraw.Draw(im)
    def txt(x, y, s, size=14, color=INK, mono=False, bold=False, anchor='lm'):
        d.text((x,y),s,font=font(size,mono,bold),fill=color,anchor=anchor)
    d.rounded_rectangle((806,53,1382,387),radius=12,fill=BG,outline=EDGE,width=1)
    txt(832,78,'FROM QWEN',12,BLUE,bold=True)
    txt(832,111,'circuit',13,MUTED,mono=True)
    for i, gate in enumerate(data['request']['circuit']):
        x=920+91*i; color=CORAL if active==i else BLUE
        fill=mix(PANEL,color,.22) if i<loaded else PANEL
        d.rounded_rectangle((x,91,x+66,130),radius=5,fill=fill,
                            outline=color if i<loaded else EDGE,width=2 if active==i else 1)
        txt(x+33,111,gate if i<loaded else '···',17,color if i<loaded else MUTED,mono=True,anchor='mm')
        if i<2: txt(x+78,111,'→',17,MUTED,anchor='mm')
    # Gate-boundary values are fixed at their real logical addresses.
    txt(832,161,'AMPLITUDE MEMORY',12,MINT,bold=True)
    txt(1149,161,f'pc {pc}/3',12,MUTED,mono=True,anchor='rm')
    txt(832,193,'basis',12,MUTED,mono=True)
    txt(984,193,'Re',12,MUTED,mono=True,anchor='rm')
    txt(1101,193,'Im',12,MUTED,mono=True,anchor='rm')
    state=data['states'][pc]['amplitudes']
    changed={0,8} if active in (0,2) else {8} if active==1 else set()
    for j, address in enumerate((0,8)):
        y=228+j*43
        if flash and address in changed:
            d.rounded_rectangle((826,y-18,1124,y+18),radius=5,fill=mix(PANEL,CORAL,.25))
        txt(832,y,f'{address:06b}',15,MUTED,mono=True)
        for x, value in zip((1004,1121),state[address]):
            txt(x,y,f'{value:+.4f}',17,MINT if value else MUTED,mono=True,anchor='rm')
    # 64 logical amplitude cells: row-major address 0..63, stable throughout.
    for a, (re,imag) in enumerate(state):
        x=1155+(a%8)*16; y=187+(a//8)*14
        weight=min(1,math.hypot(re,imag))
        fill=mix(PANEL,MINT,.18+.65*weight) if weight else PANEL
        d.rectangle((x,y,x+12,y+10),fill=fill,
                    outline=CORAL if flash and a in changed else EDGE,width=1)
    txt(1217,312,'64 amplitudes',11,MUTED,mono=True,anchor='mm')
    # Layer grouping follows the assembled checkpoint order: QPU prefix then language.
    for i in range(36):
        y=119+i*5.3
        active_group = (i<4 and mode in ('gate','result')) or (i>=4 and mode=='language')
        color=CORAL if i<4 else BLUE
        d.rectangle((1330,y,1352,y+3.4),fill=color if active_group else mix(PANEL,color,.30))
    txt(1341,79,'4 QPU',12,CORAL,anchor='mm')
    d.line((1341,89,1341,114),fill=EDGE,width=1)
    txt(1341,339,'32',13,BLUE,bold=True,anchor='mm')
    txt(1341,357,'language',11,MUTED,anchor='mm')
    d.rounded_rectangle((832,329,1284,371),radius=6,fill=PANEL,
                        outline=MINT if digits else EDGE,width=1)
    txt(847,350,'P(q3 = 1)',16,INK,mono=True)
    txt(1268,350,digits if digits is not None else '—',21,MINT if digits else MUTED,
        mono=True,bold=True,anchor='rm')
    return im


def main():
    assert sha(TRACE)==EXPECTED, 'Recorded trace changed'
    data=json.loads(TRACE.read_text())
    demo=json.loads((ROOT/'evidence/added-layers/completed-demonstrations.json').read_text())['tasks'][0]
    assert data['request']==demo['request'] and data['answer']==demo['answer']
    assert [s['pc'] for s in data['states']]==[0,1,2,3]
    assert all(len(s['amplitudes'])==64 for s in data['states'])
    measured=data['answer']['probabilities']['q3_one']
    born=sum(re*re+im*im for a,(re,im) in enumerate(data['states'][-1]['amplitudes']) if a>>3&1)
    assert abs(born-measured)<1e-6
    movies={n:sha(ROOT/'assets'/n) for n in ('demo.gif','demo.mp4')}
    background=base(); frames=[]; durations=[]; labels=[]
    def add(duration,label,**kw):
        frames.append(frame(background,data,**kw));durations.append(duration);labels.append(label)
    for loaded,duration in [(0,400),(1,450),(2,450),(3,800)]:
        add(duration,f'language-{loaded}',pc=0,loaded=loaded,mode='language')
    add(500,'initialized',pc=0)
    for i in range(3):
        add(240,f'gate-{i+1}-read',pc=i,active=i,flash=True)
        add(300,f'gate-{i+1}-write',pc=i+1,active=i,flash=True)
        add(850,f'gate-{i+1}-hold',pc=i+1,active=i)
    number=f'{measured:.10f}'
    for i in range(1,len(number)+1):
        add(80,'probability-digits',pc=3,mode='result',digits=number[:i])
    add(2300,'answer',pc=3,mode='result',digits=number)
    assert all(im.crop((0,0,806,440)).tobytes()==background.crop((0,0,806,440)).tobytes() for im in frames)
    # One palette for the whole animation avoids flicker in static title text.
    palette_sheet=Image.new('RGB',(1440,440*len(frames)))
    for i,im in enumerate(frames):palette_sheet.paste(im,(0,i*440))
    palette=palette_sheet.quantize(colors=256)
    indexed=[im.quantize(palette=palette,dither=Image.Dither.NONE) for im in frames]
    target=ROOT/'assets/banner-added-layers.gif'
    indexed[0].save(target,save_all=True,append_images=indexed[1:],duration=durations,
                    loop=0,disposal=1,optimize=False)
    frames[-1].save(ROOT/'assets/banner-added-layers-preview.png')
    check=Image.open(target); decoded=[]
    for i in range(check.n_frames):
        check.seek(i);decoded.append(check.convert('RGB').crop((0,0,806,440)).tobytes())
    assert all(x==decoded[0] for x in decoded), 'Static title moved or flickered'
    assert movies=={n:sha(ROOT/'assets'/n) for n in movies}
    review=ROOT/'review/banner-animation';review.mkdir(parents=True,exist_ok=True)
    selected=[0,3,7,10,13,len(frames)-1]
    sheet=Image.new('RGB',(1440,440*3),BG)
    for j,i in enumerate(selected):
        thumb=frames[i].resize((720,220),Image.Resampling.LANCZOS)
        sheet.paste(thumb,((j%2)*720,(j//2)*440+35))
        ImageDraw.Draw(sheet).text(((j%2)*720+25,(j//2)*440+15),labels[i],fill=MUTED,font=font(13))
    sheet.save(review/'contact-sheet.png')
    meta={'generator':'tools/render_banner_animation.py','trace_path':str(TRACE.relative_to(ROOT)),
          'trace_sha256':sha(TRACE),'source_execution_sha256':data['source_sha256'],
          'design_reference':'https://divagr.com/qwenputer/demo.gif',
          'logical_state_grid':'8 by 8, row-major amplitude addresses 0..63',
          'numerical_values':'Recorded native output digits decoded at gate boundaries',
          'layer_strip':'Schematic group roles, not physical activation telemetry',
          'timing':'Authored presentation schedule',
          'size':[1440,440],'duration_ms':sum(durations),'frames':check.n_frames,
          'probability':measured,'reconstructed_born_check':born,
          'static_left_verified':True,'preserved_animation_sha256':movies,
          'artifacts':{str(target.relative_to(ROOT)):sha(target),
                       'assets/banner-added-layers-preview.png':sha(ROOT/'assets/banner-added-layers-preview.png')}}
    (ROOT/'assets/banner-animation-metadata.json').write_text(json.dumps(meta,indent=2)+'\n')
    print(json.dumps({'gif_bytes':target.stat().st_size,'frames':check.n_frames,'duration_ms':sum(durations),
                      'old_example_preserved':True,'trace_verified':True}))


if __name__=='__main__':main()
