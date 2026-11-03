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
          svg_text(48,230,"QPU inside a language model",28),
          svg_text(48,275,"H / X / T / CNOT · state-vector execution",20,MUTED),
          f'<rect x="48" y="320" width="456" height="38" rx="7" fill="{PANEL}" stroke="{EDGE}"/>',
          svg_text(63,345,"FIXED GATES · SAMPLED COUNTS",14,MINT,weight="600")]
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

