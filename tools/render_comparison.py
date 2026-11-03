"""Original circuit/probability comparison from the frozen C16-4 evidence."""
from pathlib import Path
import hashlib
import itertools
import json
import subprocess
from PIL import Image, ImageDraw
from render_support import font, label, wrap, card, static_assets, BG, PANEL, EDGE, WHITE, MUTED, MINT, AMBER

ROOT = Path(__file__).resolve().parents[1]
W, H, FPS, SECONDS = 1280, 720, 12, 30
EVENTS = ["P(q14 = 1)", "P(q14 xor q8 xor q10 xor q7 = 1)"]


def position(t):
    if t < 3.5: return 0, 0
    if t < 9.5: return 1, min(80, int((t - 3.5) / 6 * 80))
    if t < 11: return 1, 80
    if t < 16: return 1, min(161, 80 + int((t - 11) / 5 * 81))
    if t < 22: return 2, 161
    return 3, 161


def render_frame(t, trace, cumulative):
    im = Image.new("RGB", (W, H), BG); d = ImageDraw.Draw(im)
    for x in range(36, W, 32):
        for y in range(28, H, 32): d.point((x, y), fill=EDGE)
    qx = 36 + d.textbbox((0, 0), "Qwen", font=font(49, "display"))[2] + 6
    label(d, (36, 20), "Qwen", 49, WHITE, "display"); label(d, (qx, 20), "QPU", 49, MINT, "display")
    label(d, (285, 34), "QUANTUM CIRCUITS / MEASUREMENT PROBABILITIES", 18, MUTED)
    stage, step = position(t)
    for i, text in enumerate(["01 / TASK", "02 / CIRCUIT", "03 / SAMPLES", "04 / COMPARE"]):
        x = 36 + 304 * i
        d.line((x, 111, x + 274, 111), fill=MINT if i <= stage else EDGE, width=3)
        label(d, (x, 81), text, 16, MINT if i == stage else MUTED, "mono")
    if stage == 3:
        label(d, (36, 143), "C16-4  /  MEASUREMENT PROBABILITIES", 18, MUTED, "mono")
        groups = [("Qwen", trace["plain_answer"], AMBER),
                  ("QwenQPU", trace["qwenqpu_answer"], MINT),
                  ("Qiskit reference", trace["reference_probabilities"], WHITE)]
        for j, (name, values, color) in enumerate(groups):
            x = 36 + j * 414
            card(d, (x, 183, x + 386, 607), outline=color)
            label(d, (x + 22, 204), name, 30, color)
            for k in range(2):
                y = 270 + k * 131
                wrap(d, EVENTS[k], (x + 22, y), 340, 17, MUTED, leading=23)
                value = values[f"o{k+1}"]
                label(d, (x + 22, y + 45), f"{value:.6f}", 43, color, "display")
                if j == 1:
                    one = trace["event_counts"][f"o{k+1}"]["1"]
                    label(d, (x + 224, y + 65), f"{one}/4096", 16, MUTED, "mono")
            if j < 2:
                ok = all(abs(values[f"o{k+1}"] - trace["reference_probabilities"][f"o{k+1}"]) <= .03270826686567064 for k in range(2))
                label(d, (x + 22, 562), "CORRECT" if ok else "FIRST PROBABILITY INCORRECT", 15, color, "mono")
            else: label(d, (x + 22, 562), "INDEPENDENT STATE VECTOR", 15, MUTED, "mono")
    else:
        card(d, (36, 142, 390, 606)); card(d, (410, 142, 1244, 606))
        label(d, (56, 161), "CALCULATE", 15, MINT, "mono")
        wrap(d, "Start in |0>^16. Apply the given 161-gate circuit. Return:", (56, 194), 310, 23, leading=30)
        wrap(d, EVENTS[0], (56, 310), 310, 22, WHITE, leading=29, kind="mono")
        wrap(d, EVENTS[1], (56, 365), 310, 20, WHITE, leading=28, kind="mono")
        label(d, (56, 489), "39 H   38 X   38 T", 18, MUTED, "mono")
        label(d, (56, 524), "46 CNOT", 18, MUTED, "mono")
        label(d, (56, 570), "16 qubits / 4096 shots", 18, MINT)
        if stage < 2:
            label(d, (432, 161), "CIRCUIT EXECUTION", 15, MINT, "mono")
            label(d, (1220, 161), f"{step:03}/161", 17, MUTED, "mono", anchor="ra")
            begin = max(0, min(137, step - 12)); window = trace["operations"][begin:begin + 24]
            x0, dx, y0, dy = 503, 30.4, 216, 16
            for q in range(16):
                y = y0 + q * dy
                label(d, (432, y - 8), f"q{q:02}", 12, MUTED, "mono")
                d.line((486, y, 1221, y), fill=EDGE)
            for i, op in enumerate(window):
                idx = begin + i; x = x0 + i * dx; a = op.split()
                color = AMBER if idx == step - 1 else MINT if idx < step else "#3a5261"
                if a[0] == "CNOT":
                    ya, yb = y0 + int(a[1]) * dy, y0 + int(a[2]) * dy
                    d.line((x, ya, x, yb), fill=color, width=2); d.ellipse((x - 3, ya - 3, x + 3, ya + 3), fill=color)
                    d.ellipse((x - 6, yb - 6, x + 6, yb + 6), outline=color, width=2)
                    d.line((x - 4, yb, x + 4, yb), fill=color); d.line((x, yb - 4, x, yb + 4), fill=color)
                else:
                    y = y0 + int(a[1]) * dy
                    d.rounded_rectangle((x - 9, y - 9, x + 9, y + 9), radius=2, fill=color)
                    label(d, (x, y), a[0], 12, BG, "mono", anchor="mm")
            op = trace["frames"][step]["operation"]
            label(d, (432, 482), f"{step:03}  {op}", 17, WHITE, "mono")
            if 9.5 <= t < 11:
                label(d, (1220, 482), "CONTINUE: GATES 81-161", 15, AMBER, "mono", anchor="ra")
            for j, value in enumerate(trace["frames"][step]["probabilities"]):
                y = 522 + j * 34
                label(d, (432, y), f"P{j+1}", 16, MUTED, "mono")
                d.rounded_rectangle((479, y + 3, 1010, y + 22), radius=2, fill=EDGE)
                if value > 0: d.rounded_rectangle((479, y + 3, 479 + 531 * value, y + 22), radius=2, fill=MINT if j == 0 else AMBER)
                label(d, (1217, y), f"{value:.6f}", 17, WHITE, "mono", anchor="ra")
        else:
            shots = min(4096, max(0, int((t - 16) / 5.5 * 4096)))
            label(d, (432, 163), "TERMINAL SAMPLES", 16, MINT, "mono")
            label(d, (432, 211), f"{shots:04} / 4096", 41, WHITE, "display")
            for j in range(2):
                y = 308 + j * 119; one = cumulative[j][shots]
                label(d, (432, y), EVENTS[j], 19, MUTED)
                d.rounded_rectangle((432, y + 36, 1065, y + 66), radius=3, fill=EDGE)
                if shots and one: d.rounded_rectangle((432, y + 36, 432 + 633 * one / shots, y + 66), radius=3, fill=MINT if j == 0 else AMBER)
                label(d, (1215, y + 36), f"{one}/{shots}", 21, WHITE, "mono", anchor="ra")
            label(d, (432, 574), "seed 8311126", 16, MUTED, "mono")
    label(d, (36, 649), "16 QUBITS / 161 GATES / 4096 SHOTS", 16, MINT, "mono")
    label(d, (1244, 649), "C16-4", 16, MUTED, "mono", anchor="ra")
    return im


def main():
    static_assets(); assets = ROOT / "assets"; review = ROOT / "review"; review.mkdir(exist_ok=True)
    trace = json.loads((assets / "demo-trace.json").read_text())
    cumulative = [[0] + list(itertools.accumulate(seq)) for seq in trace["event_bits"]]
    for t in [0., 6., 10., 14., 19., 25., 29.9]: render_frame(t, trace, cumulative).save(review / f"comparison-{t:04.1f}.png")
    render_frame(25., trace, cumulative).save(assets / "demo-poster.png")
    command = ["ffmpeg", "-hide_banner", "-loglevel", "error", "-y", "-f", "rawvideo", "-pix_fmt", "rgb24", "-s", "1280x720", "-r", str(FPS), "-i", "-", "-an", "-c:v", "libx264", "-preset", "medium", "-crf", "20", "-pix_fmt", "yuv420p", "-movflags", "+faststart", str(assets / "demo.mp4")]
    p = subprocess.Popen(command, stdin=subprocess.PIPE)
    for i in range(FPS * SECONDS): p.stdin.write(render_frame(i / FPS, trace, cumulative).tobytes())
    p.stdin.close(); assert p.wait() == 0
    subprocess.run(["ffmpeg", "-hide_banner", "-loglevel", "error", "-y", "-i", str(assets / "demo.mp4"), "-filter_complex", "fps=12,scale=1024:-1:flags=lanczos,split[a][b];[a]palettegen=stats_mode=diff[p];[b][p]paletteuse=dither=bayer:bayer_scale=3", "-loop", "0", str(assets / "demo.gif")], check=True)
    provenance = {"media_revision": "github-v3-C16-4-comparison", "question_id": "C16-4", "source": "frozen eval40 v4", "original_artwork": True, "new_language_generation": False, "core_trace_matches_recorded_counts": True, "segmented_80_plus_81_state_identical": True, "duration_seconds": SECONDS, "fps": FPS, "frames": SECONDS * FPS, "dimensions": [W, H], "timing": "illustrative presentation timing, no wall-clock performance shown", "trace_sha256": hashlib.sha256((assets / "demo-trace.json").read_bytes()).hexdigest(), "renderer_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(), "source_record_sha256": trace["source_record_sha256"]}
    (assets / "animation-provenance.json").write_text(json.dumps(provenance, indent=2) + "\n")
    print(json.dumps(provenance))


if __name__ == "__main__": main()
