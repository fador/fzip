#!/usr/bin/env python3
"""Generate SVG and multi-resolution Windows .ico icons for fzip / fzip-gui."""
from __future__ import annotations

import math
from pathlib import Path
from PIL import Image, ImageDraw, ImageFont

ASSETS_DIR = Path(__file__).resolve().parent
SIZES = [16, 24, 32, 48, 64, 128, 256]


def lerp_color(c1: tuple[int, int, int, int], c2: tuple[int, int, int, int], t: float) -> tuple[int, int, int, int]:
    t = max(0.0, min(1.0, t))
    return tuple(int(round(a + (b - a) * t)) for a, b in zip(c1, c2))  # type: ignore[return-value]


def make_gradient_card(size: int, c_top: tuple[int, int, int, int], c_bot: tuple[int, int, int, int],
                       b_start: tuple[int, int, int, int], b_end: tuple[int, int, int, int]) -> Image.Image:
    s = size * 4
    pad = max(4, int(s * 0.055))
    radius = int(s * 0.21)
    border_w = max(4, int(s * 0.055))

    # Diagonal gradient mask for card fill & border
    mask = Image.new("L", (s, s), 0)
    mdraw = ImageDraw.Draw(mask)
    mdraw.rounded_rectangle([pad, pad, s - pad - 1, s - pad - 1], radius=radius, fill=255)

    inner_pad = pad + border_w
    inner_r = max(2, radius - border_w)
    inner_mask = Image.new("L", (s, s), 0)
    idraw = ImageDraw.Draw(inner_mask)
    idraw.rounded_rectangle([inner_pad, inner_pad, s - inner_pad - 1, s - inner_pad - 1],
                            radius=inner_r, fill=255)

    card = Image.new("RGBA", (s, s), (0, 0, 0, 0))
    px = card.load()
    for y in range(s):
        ty = y / max(1, s - 1)
        bg = lerp_color(c_top, c_bot, ty)
        for x in range(s):
            td = (x + y) / max(1, 2 * (s - 1))
            bd = lerp_color(b_start, b_end, td)
            ma = mask.getpixel((x, y))
            ia = inner_mask.getpixel((x, y))
            if ma == 0:
                continue
            if ia == 255:
                px[x, y] = bg
            elif ia == 0:
                px[x, y] = (bd[0], bd[1], bd[2], ma)
            else:
                k = ia / 255.0
                col = lerp_color(bd, bg, k)
                px[x, y] = (col[0], col[1], col[2], ma)
    return card


def draw_zipper_spine(draw: ImageDraw.ImageDraw, s: int, accent: tuple[int, int, int, int],
                      tab_fill: tuple[int, int, int, int], x_center: float, y_top: float, y_bot: float) -> None:
    spine_w = max(2, int(s * 0.022))
    draw.rectangle([x_center - spine_w, y_top, x_center + spine_w, y_bot], fill=(42, 54, 72, 230))

    # Alternating metallic zipper teeth
    tooth_h = max(3, int(s * 0.044))
    tooth_w = max(4, int(s * 0.068))
    step = tooth_h * 2
    y = y_top + tooth_h * 0.35
    toggle = True
    while y + tooth_h < y_bot - int(s * 0.14):
        if toggle:
            draw.rounded_rectangle(
                [x_center - tooth_w - spine_w * 0.3, y, x_center + spine_w * 0.4, y + tooth_h],
                radius=max(1, tooth_h // 3),
                fill=(185, 202, 224, 245),
            )
        else:
            draw.rounded_rectangle(
                [x_center - spine_w * 0.4, y, x_center + tooth_w + spine_w * 0.3, y + tooth_h],
                radius=max(1, tooth_h // 3),
                fill=(136, 156, 184, 245),
            )
        toggle = not toggle
        y += step

    # Zipper slider + pull ring at lower portion
    slider_y = y_bot - int(s * 0.18)
    sw = max(6, int(s * 0.088))
    sh = max(6, int(s * 0.092))
    draw.rounded_rectangle(
        [x_center - sw, slider_y - sh, x_center + sw, slider_y + sh],
        radius=max(2, sw // 3),
        fill=accent,
        outline=(235, 244, 255, 255),
        width=max(2, int(s * 0.012)),
    )
    pw = max(4, int(s * 0.056))
    ph = max(6, int(s * 0.095))
    draw.rounded_rectangle(
        [x_center - pw, slider_y + sh * 0.35, x_center + pw, slider_y + sh * 0.35 + ph],
        radius=max(2, pw // 2),
        fill=tab_fill,
        outline=(235, 244, 255, 230),
        width=max(1, int(s * 0.01)),
    )


def draw_fz_monogram(draw: ImageDraw.ImageDraw, s: int, variant: str) -> None:
    # Draw crisp geometric "FZ" on right two-thirds so it remains sharp at any DPI
    font = None
    for candidate in [
        r"C:\Windows\Fonts\segoeuib.ttf",
        r"C:\Windows\Fonts\arialbd.ttf",
        r"C:\Windows\Fonts\calibrib.ttf",
    ]:
        if Path(candidate).exists():
            font_size = int(s * (0.37 if variant == "app" else 0.23))
            font = ImageFont.truetype(candidate, font_size)
            break

    if font is not None:
        if variant == "app":
            draw.text((int(s * 0.43), int(s * 0.25)), "F", font=font, fill=(88, 166, 255, 255))
            draw.text((int(s * 0.61), int(s * 0.42)), "Z", font=font, fill=(188, 140, 255, 255))
        elif variant == "compress":
            draw.text((int(s * 0.49), int(s * 0.15)), "FZ", font=font, fill=(175, 212, 255, 245))
        else:
            draw.text((int(s * 0.49), int(s * 0.15)), "FZ", font=font, fill=(165, 245, 210, 245))


def draw_badge_arrow(draw: ImageDraw.ImageDraw, s: int, kind: str) -> None:
    # High-contrast bottom-right action badge clearly readable even at 16x16
    cx = int(s * 0.67)
    cy = int(s * 0.67)
    r = int(s * 0.255)
    if kind == "compress":
        bg = (22, 102, 232, 255)
        ring = (135, 200, 255, 255)
    else:
        bg = (16, 158, 96, 255)
        ring = (145, 250, 200, 255)

    draw.ellipse([cx - r, cy - r, cx + r, cy + r], fill=bg, outline=ring, width=max(3, int(s * 0.025)))

    aw = int(r * 0.26)
    ah = int(r * 0.56)
    head_w = int(r * 0.62)
    head_h = int(r * 0.44)
    white = (248, 251, 255, 255)

    if kind == "compress":
        # Downward arrow into base tray (compression into archive)
        draw.rounded_rectangle([cx - aw, cy - ah, cx + aw, cy + int(r * 0.08)], radius=max(2, aw // 2), fill=white)
        draw.polygon(
            [(cx - head_w, cy - int(r * 0.02)), (cx + head_w, cy - int(r * 0.02)), (cx, cy + int(r * 0.50))],
            fill=white,
        )
        bar_y = cy + int(r * 0.56)
        draw.rounded_rectangle(
            [cx - int(r * 0.52), bar_y - max(3, aw // 2), cx + int(r * 0.52), bar_y + max(3, aw // 2)],
            radius=max(2, aw // 2),
            fill=white,
        )
    else:
        # Upward arrow out of base tray (extraction)
        draw.rounded_rectangle([cx - aw, cy - int(r * 0.08), cx + aw, cy + int(r * 0.52)], radius=max(2, aw // 2), fill=white)
        draw.polygon(
            [(cx - head_w, cy + int(r * 0.04)), (cx + head_w, cy + int(r * 0.04)), (cx, cy - int(r * 0.56))],
            fill=white,
        )
        bar_y = cy + int(r * 0.58)
        draw.rounded_rectangle(
            [cx - int(r * 0.52), bar_y - max(3, aw // 2), cx + int(r * 0.52), bar_y + max(3, aw // 2)],
            radius=max(2, aw // 2),
            fill=white,
        )


def render_variant(size: int, variant: str) -> Image.Image:
    s = size * 4
    if variant == "extract":
        card = make_gradient_card(
            size,
            c_top=(12, 24, 24, 255),
            c_bot=(16, 36, 34, 255),
            b_start=(32, 201, 151, 255),
            b_end=(56, 189, 248, 255),
        )
        accent = (16, 185, 129, 255)
        tab_fill = (52, 211, 153, 255)
    elif variant == "compress":
        card = make_gradient_card(
            size,
            c_top=(13, 17, 23, 255),
            c_bot=(23, 33, 52, 255),
            b_start=(47, 129, 247, 255),
            b_end=(188, 140, 255, 255),
        )
        accent = (47, 129, 247, 255)
        tab_fill = (121, 192, 255, 255)
    else:
        card = make_gradient_card(
            size,
            c_top=(13, 17, 23, 255),
            c_bot=(24, 35, 56, 255),
            b_start=(88, 166, 255, 255),
            b_end=(188, 140, 255, 255),
        )
        accent = (88, 166, 255, 255)
        tab_fill = (188, 140, 255, 255)

    draw = ImageDraw.Draw(card)
    x_zip = s * 0.27
    y_top = s * 0.14
    y_bot = s * 0.85
    draw_zipper_spine(draw, s, accent, tab_fill, x_zip, y_top, y_bot)
    draw_fz_monogram(draw, s, variant)
    if variant in ("compress", "extract"):
        draw_badge_arrow(draw, s, variant)

    return card.resize((size, size), Image.Resampling.LANCZOS)


def save_ico(variant: str, path: Path) -> None:
    # Render each exact frame from 4x supersampled geometry, saving largest as root
    # with append_images for Pillow multi-frame ICO support.
    frames = [render_variant(sz, variant) for sz in SIZES]
    frames[-1].save(
        path,
        format="ICO",
        sizes=[(sz, sz) for sz in SIZES],
        append_images=frames[:-1],
    )


def write_svg(variant: str, path: Path) -> None:
    if variant == "extract":
        g1, g2 = "#20c997", "#38bdf8"
        bg1, bg2 = "#0c1818", "#102422"
        badge = """
  <circle cx="172" cy="172" r="62" fill="#109e60" stroke="#91fac8" stroke-width="6"/>
  <rect x="159" y="160" width="26" height="40" rx="5" fill="#f8fbff"/>
  <polygon points="136,170 208,170 172,126" fill="#f8fbff"/>
  <rect x="140" y="204" width="64" height="10" rx="5" fill="#f8fbff"/>"""
        mono = '<text x="128" y="92" font-family="Segoe UI, Arial, sans-serif" font-weight="800" font-size="56" fill="#a5f5d2">FZ</text>'
    elif variant == "compress":
        g1, g2 = "#2f81f7", "#bc8cff"
        bg1, bg2 = "#0d1117", "#172134"
        badge = """
  <circle cx="172" cy="172" r="62" fill="#1666e8" stroke="#87c8ff" stroke-width="6"/>
  <rect x="159" y="130" width="26" height="38" rx="5" fill="#f8fbff"/>
  <polygon points="136,162 208,162 172,202" fill="#f8fbff"/>
  <rect x="140" y="204" width="64" height="10" rx="5" fill="#f8fbff"/>"""
        mono = '<text x="128" y="92" font-family="Segoe UI, Arial, sans-serif" font-weight="800" font-size="56" fill="#afd4ff">FZ</text>'
    else:
        g1, g2 = "#58a6ff", "#bc8cff"
        bg1, bg2 = "#0d1117", "#182338"
        badge = ""
        mono = """
  <text x="110" y="142" font-family="Segoe UI, Arial, sans-serif" font-weight="800" font-size="94" fill="#58a6ff">F</text>
  <text x="156" y="190" font-family="Segoe UI, Arial, sans-serif" font-weight="800" font-size="94" fill="#bc8cff">Z</text>"""

    teeth = []
    for k in range(7):
        y = 42 + k * 21
        x = 50 if k % 2 == 0 else 66
        col = "#b9cae0" if k % 2 == 0 else "#889cb8"
        teeth.append(f'  <rect x="{x}" y="{y}" width="20" height="10" rx="3" fill="{col}"/>')
    teeth_svg = "\n".join(teeth)

    svg = f"""<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 256 256">
  <defs>
    <linearGradient id="bg" x1="0" y1="0" x2="0" y2="1">
      <stop offset="0%" stop-color="{bg1}"/>
      <stop offset="100%" stop-color="{bg2}"/>
    </linearGradient>
    <linearGradient id="bd" x1="0" y1="0" x2="1" y2="1">
      <stop offset="0%" stop-color="{g1}"/>
      <stop offset="100%" stop-color="{g2}"/>
    </linearGradient>
  </defs>
  <rect x="14" y="14" width="228" height="228" rx="48" fill="url(#bg)" stroke="url(#bd)" stroke-width="14"/>
  <rect x="64" y="34" width="10" height="184" rx="4" fill="#2a3648"/>
{teeth_svg}
  <rect x="49" y="160" width="40" height="34" rx="8" fill="{g1}" stroke="#ebf4ff" stroke-width="3"/>
  <rect x="57" y="188" width="24" height="26" rx="6" fill="{g2}" stroke="#ebf4ff" stroke-width="2.5"/>{mono}{badge}
</svg>
"""
    path.write_text(svg, encoding="utf-8")


def main() -> None:
    for variant in ("app", "compress", "extract"):
        ico_path = ASSETS_DIR / f"fzip_{variant}.ico"
        svg_path = ASSETS_DIR / f"fzip_{variant}.svg"
        save_ico(variant, ico_path)
        write_svg(variant, svg_path)
        with Image.open(ico_path) as im:
            print(f"Generated {ico_path.name} ({ico_path.stat().st_size} B) sizes={sorted(im.ico.sizes())}")
        print(f"Generated {svg_path.name} ({svg_path.stat().st_size} B)")


if __name__ == "__main__":
    main()
