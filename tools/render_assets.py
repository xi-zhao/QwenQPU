"""Original QwenQPU artwork + evidence-driven replay. No model/network calls.

Pillow renders frames; installed ffmpeg encodes MP4/GIF. Fonts are resolved
locally and are not redistributed. All numbers come from assets/demo-trace.json.
"""
from pathlib import Path
import argparse
import collections
import functools
import hashlib
import html
import json
import math
import os
import subprocess
import sys
from PIL import Image, ImageDraw, ImageFont

ROOT = Path(__file__).resolve().parents[1]
ASSETS = ROOT / "assets"
os.environ["XDG_CACHE_HOME"] = str(ROOT / "review/cache")
Path(os.environ["XDG_CACHE_HOME"]).mkdir(parents=True, exist_ok=True)
W, H, FPS, SECONDS = 1280, 720, 12, 24
BG, PANEL, EDGE = "#0b1219", "#101d28", "#233744"
WHITE, MUTED, MINT, AMBER = "#edf3f5", "#92aab7", "#63e5d3", "#ffae69"
DISPLAY = "/System/Library/Fonts/Supplemental/DIN Condensed Bold.ttf"
BODY = "/System/Library/Fonts/Avenir Next.ttc"
MONO = "/System/Library/Fonts/Menlo.ttc"


@functools.lru_cache(maxsize=64)
def font(size, kind="body"):
    p = {"display": DISPLAY, "body": BODY, "mono": MONO}[kind]
    try:
        return ImageFont.truetype(p, size, index=7 if kind=="body" else 0)
    except OSError:
        from matplotlib.font_manager import findfont
        return ImageFont.truetype(findfont("DejaVu Sans Mono" if kind=="mono" else "DejaVu Sans"), size)


def label(d, xy, value, size=18, color=WHITE, kind="body", anchor=None):
    d.text(xy, str(value), font=font(size,kind), fill=color, anchor=anchor)


def wrap(d, text, xy, width, size=19, color=WHITE, leading=27, kind="body"):
    words = text.split();line="";y=xy[1]
    for word in words:
        candidate=(line+" "+word).strip()
        if d.textbbox((0,0),candidate,font=font(size,kind))[2]>width and line:
            label(d,(xy[0],y),line,size,color,kind);y+=leading;line=word
        else:line=candidate
    if line:label(d,(xy[0],y),line,size,color,kind)
    return y+leading


def card(d, rect, fill=PANEL, outline=EDGE, radius=14):
    d.rounded_rectangle(rect, radius=radius, fill=fill, outline=outline, width=1)


def svg_text(x,y,text,size=18,color=WHITE,family="Avenir Next,DejaVu Sans,sans-serif",weight="400"):
    return f'<text x="{x}" y="{y}" fill="{color}" font-family="{family}" font-size="{size}" font-weight="{weight}">{html.escape(text)}</text>'


def static_assets():
    if sys.platform=="darwin" and Path("/opt/homebrew/lib/libcairo.2.dylib").is_file():
        # Only this render process; use the existing Homebrew Cairo library.
        os.environ.setdefault("DYLD_FALLBACK_LIBRARY_PATH", "/opt/homebrew/lib")
    import cairosvg
    # All paths, typography, colors and composition are original. No reference
    # logo, frames, grids or source code are imported into these assets.
    s=[f'<svg xmlns="http://www.w3.org/2000/svg" width="1280" height="410" viewBox="0 0 1280 410"><rect width="1280" height="410" fill="{BG}"/>']
    for x in range(24,1280,32):
        for y in range(24,410,32):s.append(f'<circle cx="{x}" cy="{y}" r=".6" fill="{EDGE}"/>')
    s += [svg_text(48,61,"LANGUAGE / FIXED QUANTUM NETWORK",15,MUTED,weight="600"),
          f'<text x="43" y="174" fill="{WHITE}" font-family="DIN Condensed,DejaVu Sans Condensed,sans-serif" font-size="100" font-weight="700">Qwen<tspan fill="{MINT}">QPU</tspan></text>',
          svg_text(48,230,"A language model. A fixed quantum network.",28),
          svg_text(48,275,"1–16 qubits · H / X / T / CNOT · no training",20,MUTED),
          f'<rect x="48" y="320" width="456" height="38" rx="7" fill="{PANEL}" stroke="{EDGE}"/>',
          svg_text(63,345,"IDEAL CLASSICAL SIMULATION · HOST SCHEDULED",14,MINT,weight="600")]
    for i,y in enumerate([128,202,276]):
        s.append(svg_text(785,y+6,f"q{i}",16,MUTED,"Menlo,monospace"))
        s.append(f'<path d="M830 {y} H1215" stroke="{EDGE}" stroke-width="2"/>')
    s += [f'<rect x="864" y="106" width="44" height="44" rx="7" fill="{MINT}"/>',svg_text(876,136,"H",25,BG,weight="700")]
    for x,a,b in [(978,128,202),(1108,202,276)]:
        s += [f'<path d="M{x} {a} V{b}" stroke="{MINT}" stroke-width="2"/>',
              f'<circle cx="{x}" cy="{a}" r="5" fill="{MINT}"/>',
              f'<circle cx="{x}" cy="{b}" r="13" fill="{BG}" stroke="{MINT}" stroke-width="2"/>',
              f'<path d="M{x-8} {b} H{x+8} M{x} {b-8} V{b+8}" stroke="{MINT}" stroke-width="2"/>']
    s += [svg_text(832,337,"COMPUTE THE CIRCUIT.",20,MINT,"Menlo,monospace"),
          svg_text(832,367,"REPORT THE COUNTS.",20,AMBER,"Menlo,monospace"),"</svg>"]
    (ASSETS/"banner.svg").write_text("\n".join(s))
    cairosvg.svg2png(bytestring="\n".join(s).encode(),write_to=str(ASSETS/"banner.png"))

    # Actual implementation boundary, not a diagram of future decoder fusion.
    s=[f'<svg xmlns="http://www.w3.org/2000/svg" width="1400" height="650" viewBox="0 0 1400 650"><defs><marker id="arrow" markerWidth="8" markerHeight="8" refX="6" refY="3" orient="auto"><path d="M0 0 L6 3 L0 6" fill="none" stroke="{MINT}"/></marker></defs><rect width="1400" height="650" fill="{BG}"/>',
       svg_text(46,63,"THE EXECUTION PATH",38,WHITE,"DIN Condensed,DejaVu Sans Condensed,sans-serif","700"),
       svg_text(46,100,"Explicit host scheduling. A separate quantum state. No decoder / KV fusion.",21,MUTED)]
    boxes=[(46,190,225,148,"Qwen","language.generate","Propose circuit JSON",AMBER),
           (307,190,211,148,"Host","validate + encode","Supply seeded uniforms",MUTED),
           (554,151,330,266,"Fixed quantum module","QwenQPU.forward","TorchScript control",MINT),
           (920,190,204,148,"Host","aggregate bits","Return counts only",MUTED),
           (1160,190,194,148,"Qwen","language.generate","Report the counts",AMBER)]
    for x,y,w,h,title,line1,line2,c in boxes:
        s += [f'<rect x="{x}" y="{y}" width="{w}" height="{h}" rx="14" fill="{PANEL}" stroke="{c}"/>',
              svg_text(x+18,y+37,title,24,c,weight="600"),svg_text(x+18,y+73,line1,16,WHITE,"Menlo,monospace"),
              svg_text(x+18,y+104,line2,16,MUTED)]
    s += [svg_text(574,301,"address → local 4×4 linear",17,WHITE,"Menlo,monospace"),
          svg_text(574,333,"mask → state writeback",17,WHITE,"Menlo,monospace"),
          svg_text(574,365,"CDF → terminal sampling",17,WHITE,"Menlo,monospace"),
          svg_text(574,397,"FP32 state: [2^n, 2]",17,MINT,"Menlo,monospace")]
    for a,b in [(271,307),(518,554),(884,920),(1124,1160)]:s.append(f'<path d="M{a+4} 264 H{b-9}" stroke="{MINT}" stroke-width="2" marker-end="url(#arrow)"/>')
    s += [f'<path d="M46 468 H1354" stroke="{EDGE}"/>',
          svg_text(46,512,"NETWORK",16,MINT,weight="600"),svg_text(180,512,"Gate selection, state updates and bitstring selection happen in the scripted module.",20),
          svg_text(46,552,"HOST",16,AMBER,weight="600"),svg_text(180,552,"Parsing, repeat expansion, RNG and counts aggregation remain in the driver.",20),
          svg_text(46,604,"Qiskit is an offline test oracle. No external simulator executes production circuits.",18,MUTED),"</svg>"]
    (ASSETS/"architecture.svg").write_text("\n".join(s))
    cairosvg.svg2png(bytestring="\n".join(s).encode(),write_to=str(ASSETS/"architecture.png"))


def render_frame(t, trace, record):
    im=Image.new("RGB",(W,H),BG);d=ImageDraw.Draw(im)
    for x in range(36,W,32):
        for y in range(28,H,32):d.point((x,y),fill=EDGE)
    qx=36+d.textbbox((0,0),"Qwen",font=font(49,"display"))[2]+6
    label(d,(36,20),"Qwen",49,WHITE,"display");label(d,(qx,20),"QPU",49,MINT,"display")
    header_x=max(277,qx+d.textbbox((0,0),"QPU",font=font(49,"display"))[2]+35)
    label(d,(header_x,34),"LANGUAGE / CIRCUIT / COUNTS",19,MUTED)
    card(d,(997,27,1244,66),outline=MINT);label(d,(1010,36),"CLASSICAL SIMULATION",17,MINT,"mono")
    stage=0 if t<4.2 else 1 if t<15.5 else 2 if t<19.2 else 3
    headings=["01 / REQUEST","02 / EVOLVE","03 / SAMPLE","04 / ANSWER"]
    for i,text in enumerate(headings):
        x=36+i*304;d.line((x,111,x+274,111),fill=MINT if i<=stage else EDGE,width=3)
        label(d,(x,81),text,16,MINT if i==stage else MUTED,"mono")
    card(d,(36,142,390,606));card(d,(410,142,1244,606))
    label(d,(56,159),"TASK",14,MUTED,"mono")
    wrap(d,"Prepare a 16-qubit GHZ state from all zeros using H and a chain of CNOTs.",(56,188),314,22,leading=28)
    if stage<3:
        label(d,(56,297),"QWEN CIRCUIT",14,AMBER,"mono")
        label(d,(56,325),"16 qubits · 16 gates · 1024 shots",16,MUTED)
        ops=trace["packet"]["circuit"]
        step=max(0,min(16,int((t-4.5)/.6)+1)) if stage>=1 else 0
        begin=max(0,min(9,step-4));shown=ops[begin:begin+7]
        for i,op in enumerate(shown):
            y=360+i*28;index=begin+i
            if index==step-1 and stage==1:d.rounded_rectangle((49,y-2,378,y+25),radius=4,fill="#203e46")
            label(d,(59,y),f"{index+1:02}  {op}",18,MINT if index<step else WHITE,"mono")
    else:
        label(d,(56,297),"QWEN ANSWER",14,AMBER,"mono")
        label(d,(56,334),'"shots": 1024,',19,WHITE,"mono")
        label(d,(56,371),'"counts": {',19,WHITE,"mono")
        label(d,(64,410),'"0000000000000000": 534,',16,AMBER,"mono")
        label(d,(64,449),'"1111111111111111": 490',16,AMBER,"mono")
        label(d,(56,488),"}",19,WHITE,"mono")
    step=max(0,min(16,int((t-4.5)/.6)+1)) if stage>=1 else 0
    current=trace["frames"][step]
    if stage<2:
        label(d,(432,160),"FIXED QUANTUM NETWORK",15,MINT,"mono")
        label(d,(1222,159),f"{step:02}/16  {current['operation']}",16,MUTED,"mono",anchor="ra")
        x0,x1,y0,dy=489,1215,217,15.5
        for q in range(16):
            y=y0+q*dy;label(d,(432,y-9),f"q{q:02}",12,MUTED,"mono")
            d.line((x0-10,y,x1+9,y),fill=EDGE,width=1)
        for i,op in enumerate(trace["packet"]["circuit"]):
            x=x0+i*(x1-x0)/15;parts=op.split();past=i<step;col=MINT if past else "#3a5261"
            if parts[0]=="H":
                d.rounded_rectangle((x-11,y0-11,x+11,y0+11),radius=3,fill=col)
                label(d,(x,y0),"H",14,BG,"mono",anchor="mm")
            else:
                a,b=map(int,parts[1:]);ya,yb=y0+a*dy,y0+b*dy
                d.line((x,ya,x,yb),fill=col,width=2)
                d.ellipse((x-3,ya-3,x+3,ya+3),fill=col)
                d.ellipse((x-7,yb-7,x+7,yb+7),fill=PANEL,outline=col,width=2)
                d.line((x-5,yb,x+5,yb),fill=col,width=1);d.line((x,yb-5,x,yb+5),fill=col,width=1)
            if i==step-1 and stage==1:d.line((x,201,x,478),fill=AMBER,width=1)
        d.line((432,488,1220,488),fill=EDGE)
        label(d,(432,502),"BASIS PROBABILITIES",13,MUTED,"mono")
        for j,s in enumerate(current["sparse_state"]):
            y=532+j*28;label(d,(432,y),s["bitstring"],16,WHITE,"mono")
            d.rounded_rectangle((622,y+2,622+410*s["probability"],y+19),radius=2,fill=MINT)
            label(d,(1060,y),f"p={s['probability']:.3f}",15,MINT,"mono")
    else:
        label(d,(432,160),"TERMINAL SAMPLES / OBSERVED COUNTS",15,MINT,"mono")
        progress=min(1,max(0,(t-15.5)/3.5));n=int(progress*trace["shots"])
        counts=collections.Counter(trace["samples"][:n])
        label(d,(432,209),"Terminal measurement",28,WHITE)
        label(d,(432,249),f"{n:04} / 1024 samples",20,MUTED)
        for j,bits in enumerate(["0"*16,"1"*16]):
            y=318+j*100;count=counts[bits]
            label(d,(432,y),bits,21,WHITE,"mono")
            d.rounded_rectangle((432,y+37,1120,y+66),radius=3,fill=EDGE)
            if count:d.rounded_rectangle((432,y+37,432+688*count/600,y+66),radius=3,fill=MINT if j==0 else AMBER)
            label(d,(1205,y+44),str(count),28,MINT if j==0 else AMBER,"mono",anchor="ra")
        label(d,(432,538),"seed=123  ·  bit order q15..q0",17,MUTED,"mono")
    label(d,(36,642),"GHZ16 / H + CNOT / 1024 SHOTS",16,MINT,"mono")
    label(d,(1244,642),"DEMO REPLAY",15,MUTED,"mono",anchor="ra")
    return im


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--stills-only",action="store_true")
    args=ap.parse_args();ASSETS.mkdir(exist_ok=True)
    static_assets()
    trace=json.loads((ASSETS/"demo-trace.json").read_text())
    record=json.loads((ROOT/"evidence/public/ghz16.json").read_text())
    assert trace["counts"]==record["visible_result"]["counts"]
    for t in [1.,7.2,13.5,17.5,21.]:
        render_frame(t,trace,record).save(ROOT/"review"/f"demo-{t:04.1f}.png")
    render_frame(21.,trace,record).save(ASSETS/"demo-poster.png")
    if not args.stills_only:
        command=["ffmpeg","-hide_banner","-loglevel","error","-y","-f","rawvideo","-pix_fmt","rgb24","-s",f"{W}x{H}","-r",str(FPS),"-i","-","-an","-c:v","libx264","-preset","medium","-crf","20","-pix_fmt","yuv420p","-movflags","+faststart",str(ASSETS/"demo.mp4")]
        p=subprocess.Popen(command,stdin=subprocess.PIPE)
        for i in range(FPS*SECONDS):p.stdin.write(render_frame(i/FPS,trace,record).tobytes())
        p.stdin.close()
        if p.wait()!=0:raise RuntimeError("MP4 encoding failed")
        subprocess.run(["ffmpeg","-hide_banner","-loglevel","error","-y","-i",str(ASSETS/"demo.mp4"),"-filter_complex","fps=12,scale=1024:-1:flags=lanczos,split[a][b];[a]palettegen=stats_mode=diff[p];[b][p]paletteuse=dither=bayer:bayer_scale=3","-loop","0",str(ASSETS/"demo.gif")],check=True)
    info={"original_artwork":True,"reference_assets_copied":False,"trace_sha256":hashlib.sha256((ASSETS/"demo-trace.json").read_bytes()).hexdigest(),"renderer_sha256":hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),"frames":FPS*SECONDS,"fps":FPS,"duration_seconds":SECONDS,"dimensions":[W,H],"speed_is_illustrative":True,"physical_QPU":False,"new_language_inference":False,"source":"Recorded Qwen GHZ16 packet, unchanged core CPU replay, historical counts reproduced","media_revision":"github-v2-concise-labels","on_screen_context":["CLASSICAL SIMULATION","DEMO REPLAY"],"extended_provenance":"docs/MEDIA.md"}
    (ASSETS/"animation-provenance.json").write_text(json.dumps(info,indent=2)+"\n")
    print(json.dumps(info))


if __name__=="__main__":main()
