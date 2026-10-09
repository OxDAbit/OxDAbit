"""Generate assets/banner.svg: a terminal-style profile banner with a 1-bit dithered portrait.

Usage:
    pip install pillow numpy
    python assets/generate_banner.py

Edit PROFILE_YAML below and re-run to update the banner text.
"""

from collections import deque
from pathlib import Path
from xml.sax.saxutils import escape

import numpy as np
from PIL import Image, ImageFilter, ImageOps

HERE = Path(__file__).parent
PORTRAIT = HERE / "portrait.webp"
OUTPUT = HERE / "banner.svg"

# --- palette (dark terminal blue) -------------------------------------------
BG = "#0A0F1C"
WINDOW = "#0D1526"
PANEL = "#0F1A2E"
BORDER = "#22344F"
DIM = "#6B7A93"
LINENO = "#34445E"
KEY = "#5AA9E6"
VALUE = "#C8D3E6"
COMMENT = "#4E5D78"
AQUA = "#5EEAD4"
DOTS = "#5B9FE3"
FONT = "ui-monospace,SFMono-Regular,Menlo,Consolas,'Liberation Mono',monospace"

# --- portrait grid ------------------------------------------------------------
GRID_W, GRID_H = 190, 220   # dithered resolution
DOT = 2                     # px per dot in the SVG
CROP = (40, 20, 856, 965)   # face + shoulders crop of the source photo

# Each line is a list of (text, color) tokens.
PROFILE_YAML = [
    [("# 0xDA_bit · profile.yml", COMMENT)],
    [("profile:", KEY)],
    [("  name: ", KEY), ("David Álvarez", VALUE)],
    [("  aka: ", KEY), ("0xDA_bit", VALUE)],
    [("  role: ", KEY), ("IoT Engineer", VALUE)],
    [("  experience: ", KEY), ("15+ yrs IoT · 25+ yrs hardware", VALUE)],
    [("  education: ", KEY), ("B.Eng. Industrial Electronics Engineering", VALUE)],
    [("  location: ", KEY), ("Barcelona, Spain", VALUE)],
    [("  work_mode: ", KEY), ("[remote, hybrid]", VALUE)],
    [("  status: ", KEY), ("open_to_opportunities", AQUA)],
    [],
    [("focus:", KEY)],
    [("  iot: ", KEY), ("[ESP32, ESP-IDF, HomeKit]", VALUE)],
    [("  embedded_linux: ", KEY), ("[Industrial, Docker, Kubernetes, Raspberry Pi]", VALUE)],
    [("  domains: ", KEY), ("[Embedded HW/SW, Smart Home, AI]", VALUE)],
    [("  languages: ", KEY), ("[Bash, C, C++, Go, Java, Python, React, TypeScript]", VALUE)],
    [("  learning: ", KEY), ("[Rust]", VALUE)],
    [],
    [("links:", KEY)],
    [("  linkedin: ", KEY), ("linkedin.com/in/0xdabit", VALUE)],
    [("  youtube: ", KEY), ("youtube.com/@0xDAbitLabs", VALUE)],
    [("  web: ", KEY), ("oxdabit.com", VALUE)],
    [("  email: ", KEY), ("hello@oxdabit.com", VALUE)],
]

DESCRIPTION = (
    "David Álvarez (0xDA_bit), IoT Engineer based in Barcelona, Spain. "
    "15+ years in IoT and 25+ years in hardware. B.Eng. in Industrial Electronics Engineering. "
    "Focus: ESP32, ESP8266, ESP-IDF, Apple HomeKit, Tasmota; industrial embedded Linux with Docker, Kubernetes "
    "and SquashFS images; Raspberry Pi, embedded hardware/software, smart home and AI. "
    "Languages: Bash, C, C++, Go, Java, Python, React, TypeScript; learning Rust. "
    "Open to remote or hybrid opportunities."
)


def box_mean(a: np.ndarray, r: int) -> np.ndarray:
    k = 2 * r + 1
    c = np.pad(a, ((r + 1, r), (r + 1, r)), mode="edge").cumsum(0).cumsum(1)
    return (c[k:, k:] - c[:-k, k:] - c[k:, :-k] + c[:-k, :-k]) / (k * k)


def background_mask(img: Image.Image) -> np.ndarray:
    """Flood the smooth, bright studio background inwards from the borders.

    The background has almost no local texture, while skin, hair and the heathered
    sweater do, so 'flat and bright' + 'connected to the border' isolates it.
    """
    f = np.asarray(img, float)
    mean = box_mean(f, 4)
    std = np.sqrt(np.clip(box_mean(f * f, 4) - mean * mean, 0, None))
    flat = Image.fromarray(((std < 1.3) & (mean > 150)).astype(np.uint8) * 255)
    cand = np.asarray(flat.resize((272, 315), Image.BOX)) > 200
    h, w = cand.shape
    bg = np.zeros_like(cand)
    queue = deque()
    seeds = [(0, x) for x in range(w)] + [(y, x) for y in range(h) for x in (0, w - 1)]
    for y, x in seeds:
        if cand[y, x]:
            bg[y, x] = True
            queue.append((y, x))
    while queue:
        y, x = queue.popleft()
        for dy, dx in ((1, 0), (-1, 0), (0, 1), (0, -1)):
            ny, nx = y + dy, x + dx
            if 0 <= ny < h and 0 <= nx < w and cand[ny, nx] and not bg[ny, nx]:
                bg[ny, nx] = True
                queue.append((ny, nx))
    mask = Image.fromarray((bg * 255).astype(np.uint8)).filter(ImageFilter.MaxFilter(5))
    return np.asarray(mask.resize((GRID_W, GRID_H), Image.BILINEAR), float) / 255


def floyd_steinberg(gray: np.ndarray) -> np.ndarray:
    """Serpentine Floyd-Steinberg dithering; returns a boolean 'dot on' grid."""
    g = gray.copy()
    h, w = g.shape
    out = np.zeros_like(g, bool)
    for y in range(h):
        d = 1 if y % 2 == 0 else -1
        for x in (range(w) if d == 1 else range(w - 1, -1, -1)):
            old = g[y, x]
            new = 255.0 if old >= 128 else 0.0
            out[y, x] = new > 0
            err = old - new
            if 0 <= x + d < w:
                g[y, x + d] += err * 7 / 16
            if y + 1 < h:
                if 0 <= x - d < w:
                    g[y + 1, x - d] += err * 3 / 16
                g[y + 1, x] += err * 5 / 16
                if 0 <= x + d < w:
                    g[y + 1, x + d] += err * 1 / 16
    return out


def portrait_dots() -> np.ndarray:
    full = Image.open(PORTRAIT).convert("L").crop(CROP)
    bg = background_mask(full)
    img = full.resize((GRID_W, GRID_H), Image.LANCZOS)
    img = ImageOps.autocontrast(img, cutoff=1).filter(ImageFilter.UnsharpMask(2, 120, 2))
    a = (np.asarray(img, float) / 255) ** 1.35 * 0.92
    fade = np.clip((GRID_H - np.arange(GRID_H)) / (GRID_H * 0.30), 0.35, 1)[:, None]
    return floyd_steinberg(a * fade * (1 - bg) * 255)


def dots_to_paths(dots: np.ndarray, ox: int, oy: int, bands: int = 4) -> list[str]:
    """Encode horizontal runs of dots as path segments, split into row bands for the glitch effect."""
    paths = [[] for _ in range(bands)]
    for y in range(dots.shape[0]):
        row = dots[y]
        x = 0
        while x < len(row):
            if row[x]:
                start = x
                while x < len(row) and row[x]:
                    x += 1
                paths[(y // 11) % bands].append(
                    f"M{ox + start * DOT} {oy + y * DOT + DOT // 2}h{(x - start) * DOT}")
            else:
                x += 1
    return ["".join(p) for p in paths]


def text(x, y, content, color, size=14, weight=None, anchor=None, extra=""):
    attrs = f'x="{x}" y="{y}" fill="{color}" font-size="{size}"'
    if weight:
        attrs += f' font-weight="{weight}"'
    if anchor:
        attrs += f' text-anchor="{anchor}"'
    return f'<text {attrs}{extra}>{escape(content)}</text>'


def build_svg() -> str:
    W, H = 1200, 650
    px, py = 56, 128                          # portrait origin
    pw, ph = GRID_W * DOT, GRID_H * DOT
    code_x, code_y, step = 540, 142, 19
    reveal = 2.4                              # seconds for the portrait scan-in

    out = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{W}" height="{H}" viewBox="0 0 {W} {H}" '
        f'role="img" aria-labelledby="title desc" font-family="{FONT}">',
        '<title id="title">David Álvarez (0xDA_bit) · IoT Engineer</title>',
        f'<desc id="desc">{escape(DESCRIPTION)}</desc>',
        "<defs>",
        f'<clipPath id="reveal"><rect x="{px}" y="{py}" width="{pw}" height="0">'
        f'<animate attributeName="height" from="0" to="{ph}" dur="{reveal}s" fill="freeze"/></rect></clipPath>',
    ]
    for i, tokens in enumerate(PROFILE_YAML):
        if not tokens:
            continue
        begin = reveal * 0.3 + i * 0.11
        out.append(
            f'<clipPath id="l{i}"><rect x="{code_x}" y="{code_y + i * step - 15}" width="0" height="{step}">'
            f'<animate attributeName="width" from="0" to="620" begin="{begin:.2f}s" dur="0.35s" fill="freeze"/>'
            f'</rect></clipPath>')
    out.append("</defs>")

    # window chrome
    out += [
        f'<rect width="{W}" height="{H}" rx="18" fill="{BG}"/>',
        f'<rect x="14" y="14" width="{W - 28}" height="{H - 28}" rx="12" fill="{WINDOW}" stroke="{BORDER}"/>',
        f'<path d="M14 60H{W - 14}" stroke="{BORDER}"/>',
        '<circle cx="40" cy="37" r="6" fill="#FF5F57"/>',
        '<circle cx="61" cy="37" r="6" fill="#FEBC2E"/>',
        '<circle cx="82" cy="37" r="6" fill="#28C840"/>',
        text(W // 2, 42, "nvim ~/profile.yml — 0xDA_bit", DIM, 13, anchor="middle"),
    ]

    # left panel: visual map
    out += [
        f'<rect x="36" y="80" width="420" height="500" rx="6" fill="{PANEL}" stroke="{BORDER}"/>',
        f'<path d="M36 116H456" stroke="{BORDER}"/>',
        text(52, 103, "VISUAL.MAP", KEY, 13, weight=700, extra=' letter-spacing="1.2"'),
        text(440, 103, f"{GRID_W}×{GRID_H} · 1-BIT · FS-DITHER", DIM, 11, anchor="end"),
    ]
    corner = 12
    x0, y0, x1, y1 = px - 8, py - 6, px + pw + 8, py + ph + 6
    out.append(
        f'<path d="M{x0} {y0 + corner}V{y0}H{x0 + corner}M{x1 - corner} {y0}H{x1}V{y0 + corner}'
        f'M{x1} {y1 - corner}V{y1}H{x1 - corner}M{x0 + corner} {y1}H{x0}V{y1 - corner}" '
        f'fill="none" stroke="{KEY}" opacity=".5"/>')

    bands = dots_to_paths(portrait_dots(), px, py)
    shifts = ["3 0", "-4 0", "2 0", "-2 0"]
    out.append(f'<g clip-path="url(#reveal)" shape-rendering="crispEdges" fill="none" '
               f'stroke="{DOTS}" stroke-width="{DOT}">')
    for i, d in enumerate(bands):
        # short horizontal glitch on one band at a time, every 9 s
        b = reveal + 3 + i * 0.15
        out.append(
            f'<path d="{d}"><animateTransform attributeName="transform" type="translate" '
            f'values="0 0;{shifts[i]};0 0" keyTimes="0;0.5;1" begin="{b:.2f}s;g{i}.end+9s" '
            f'dur="0.18s" id="g{i}"/></path>')
    out.append("</g>")

    # scanline: follows the reveal, then sweeps periodically
    out.append(
        f'<rect x="{px}" y="{py}" width="{pw}" height="2" fill="{AQUA}" opacity="0">'
        f'<animate attributeName="y" values="{py};{py + ph}" dur="{reveal}s" fill="freeze"/>'
        f'<animate attributeName="opacity" values=".9;.9;0" keyTimes="0;0.95;1" dur="{reveal}s" fill="freeze"/>'
        f'</rect>'
        f'<rect x="{px}" y="{py}" width="{pw}" height="1" fill="{AQUA}" opacity="0">'
        f'<animate attributeName="y" values="{py};{py + ph}" begin="{reveal + 4}s;s2.end+6s" dur="3s" id="s2"/>'
        f'<animate attributeName="opacity" values="0;.35;.35;0" keyTimes="0;0.1;0.9;1" '
        f'begin="{reveal + 4}s;s2.begin" dur="3s"/>'
        f'</rect>')

    # right panel: profile.yml
    out += [
        f'<rect x="476" y="80" width="688" height="500" rx="6" fill="{PANEL}" stroke="{BORDER}"/>',
        f'<path d="M476 116H1164" stroke="{BORDER}"/>',
        f'<rect x="488" y="88" width="146" height="22" rx="4" fill="{WINDOW}" stroke="{BORDER}"/>',
        text(500, 103, "profile.yml", VALUE, 12),
        text(590, 103, "[yaml]", DIM, 11),
        f'<rect x="1046" y="89" width="106" height="20" rx="10" fill="{KEY}" fill-opacity=".14" stroke="{KEY}" stroke-opacity=".5"/>',
        text(1099, 103, "@0xDA_bit", KEY, 12, anchor="middle"),
    ]
    for i, tokens in enumerate(PROFILE_YAML):
        y = code_y + i * step
        out.append(text(520, y, f"{i + 1:>2}", LINENO, 12, anchor="end", extra=' xml:space="preserve"'))
        if not tokens:
            continue
        spans = "".join(f'<tspan fill="{c}">{escape(t)}</tspan>' for t, c in tokens)
        style = ' font-style="italic"' if tokens[0][1] == COMMENT else ""
        out.append(f'<text x="{code_x}" y="{y}" font-size="14" xml:space="preserve" '
                   f'clip-path="url(#l{i})"{style}>{spans}</text>')

    # blinking cursor after the last line
    last = len(PROFILE_YAML) - 1
    last_len = sum(len(t) for t, _ in PROFILE_YAML[last])
    cursor_begin = reveal * 0.3 + last * 0.11 + 0.4
    out.append(
        f'<rect x="{code_x + last_len * 8.43 + 4:.0f}" y="{code_y + last * step - 13}" width="8" height="16" '
        f'fill="{AQUA}" opacity="0"><animate attributeName="opacity" values="1;1;0;0" '
        f'keyTimes="0;0.5;0.5;1" dur="1s" begin="{cursor_begin:.2f}s" repeatCount="indefinite"/></rect>')

    # status bar
    out += [
        f'<rect x="36" y="592" width="{W - 72}" height="24" rx="4" fill="{PANEL}" stroke="{BORDER}"/>',
        f'<rect x="36" y="592" width="78" height="24" rx="4" fill="{KEY}"/>',
        text(75, 608, "NORMAL", BG, 12, weight=700, anchor="middle"),
        text(128, 608, "profile.yml", VALUE, 12),
        text(220, 608, "IoT · embedded · smart home", DIM, 12),
        text(W - 50, 608, f"yaml · utf-8 · {len(PROFILE_YAML)}L", DIM, 12, anchor="end"),
        "</svg>",
    ]
    return "\n".join(out)


if __name__ == "__main__":
    OUTPUT.write_text(build_svg(), encoding="utf-8")
    print(f"wrote {OUTPUT} ({OUTPUT.stat().st_size / 1024:.0f} KB)")
