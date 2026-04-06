"""Small local utility that renders Markdown notes into a simple PDF."""

from __future__ import annotations

import argparse
import re
from pathlib import Path
from typing import Iterable

from PIL import Image, ImageDraw, ImageFont


PAGE_SIZE = (1654, 2339)  # A4 at ~150 DPI
MARGIN_X = 120
MARGIN_Y = 120
LINE_SPACING = 10
PARAGRAPH_SPACING = 14
SECTION_SPACING = 22
TEXT_COLOR = (20, 20, 20)
BACKGROUND = (255, 255, 255)
H1_SIZE = 38
H2_SIZE = 30
H3_SIZE = 24
BODY_SIZE = 20
CODE_SIZE = 18


def load_font(size: int, bold: bool = False, mono: bool = False) -> ImageFont.FreeTypeFont | ImageFont.ImageFont:
    candidates: list[str] = []
    if mono:
        candidates.extend(
            [
                "C:/Windows/Fonts/consola.ttf",
                "C:/Windows/Fonts/cour.ttf",
            ]
        )
    elif bold:
        candidates.extend(
            [
                "C:/Windows/Fonts/arialbd.ttf",
                "C:/Windows/Fonts/calibrib.ttf",
            ]
        )
    else:
        candidates.extend(
            [
                "C:/Windows/Fonts/arial.ttf",
                "C:/Windows/Fonts/calibri.ttf",
            ]
        )

    for path in candidates:
        font_path = Path(path)
        if font_path.exists():
            return ImageFont.truetype(str(font_path), size=size)
    return ImageFont.load_default()


def wrap_text(draw: ImageDraw.ImageDraw, text: str, font: ImageFont.ImageFont, max_width: int) -> list[str]:
    if not text:
        return [""]

    lines: list[str] = []
    for paragraph in text.splitlines() or [""]:
        if not paragraph.strip():
            lines.append("")
            continue
        words = paragraph.split()
        current = words[0]
        for word in words[1:]:
            trial = f"{current} {word}"
            width = draw.textbbox((0, 0), trial, font=font)[2]
            if width <= max_width:
                current = trial
            else:
                lines.append(current)
                current = word
        lines.append(current)
    return lines


def markdown_blocks(text: str) -> Iterable[tuple[str, str]]:
    in_code = False
    for raw_line in text.splitlines():
        line = raw_line.rstrip()
        stripped = line.strip()

        if stripped.startswith("```"):
            in_code = not in_code
            continue

        if in_code:
            yield ("code", line)
            continue

        if not stripped:
            yield ("blank", "")
            continue

        if stripped.startswith("### "):
            yield ("h3", stripped[4:])
            continue
        if stripped.startswith("## "):
            yield ("h2", stripped[3:])
            continue
        if stripped.startswith("# "):
            yield ("h1", stripped[2:])
            continue
        if stripped.startswith("- "):
            yield ("bullet", stripped[2:])
            continue
        if re.match(r"^\d+\.\s", stripped):
            yield ("numbered", stripped)
            continue

        yield ("body", line)


def render_markdown_to_pdf(source: Path, destination: Path) -> None:
    text = source.read_text(encoding="utf-8")
    destination.parent.mkdir(parents=True, exist_ok=True)

    body_font = load_font(BODY_SIZE)
    bold_body_font = load_font(BODY_SIZE, bold=True)
    code_font = load_font(CODE_SIZE, mono=True)
    h1_font = load_font(H1_SIZE, bold=True)
    h2_font = load_font(H2_SIZE, bold=True)
    h3_font = load_font(H3_SIZE, bold=True)

    pages: list[Image.Image] = []
    page = Image.new("RGB", PAGE_SIZE, BACKGROUND)
    draw = ImageDraw.Draw(page)
    y = MARGIN_Y
    max_width = PAGE_SIZE[0] - (2 * MARGIN_X)
    content_bottom = PAGE_SIZE[1] - MARGIN_Y

    def new_page() -> None:
        nonlocal page, draw, y
        pages.append(page)
        page = Image.new("RGB", PAGE_SIZE, BACKGROUND)
        draw = ImageDraw.Draw(page)
        y = MARGIN_Y

    def ensure_space(height: int) -> None:
        nonlocal y
        if y + height > content_bottom:
            new_page()

    def draw_lines(lines: list[str], font: ImageFont.ImageFont, indent: int = 0) -> None:
        nonlocal y
        bbox = draw.textbbox((0, 0), "Ag", font=font)
        line_height = (bbox[3] - bbox[1]) + LINE_SPACING
        ensure_space(max(line_height * max(len(lines), 1), line_height))
        for line in lines:
            draw.text((MARGIN_X + indent, y), line, font=font, fill=TEXT_COLOR)
            y += line_height

    for block_type, value in markdown_blocks(text):
        if block_type == "blank":
            y += PARAGRAPH_SPACING
            continue

        if block_type == "h1":
            y += SECTION_SPACING
            lines = wrap_text(draw, value, h1_font, max_width)
            draw_lines(lines, h1_font)
            y += PARAGRAPH_SPACING
            continue

        if block_type == "h2":
            y += PARAGRAPH_SPACING
            lines = wrap_text(draw, value, h2_font, max_width)
            draw_lines(lines, h2_font)
            y += PARAGRAPH_SPACING
            continue

        if block_type == "h3":
            lines = wrap_text(draw, value, h3_font, max_width)
            draw_lines(lines, h3_font)
            y += PARAGRAPH_SPACING
            continue

        if block_type == "bullet":
            bullet_prefix = "\u2022 "
            lines = wrap_text(draw, value, body_font, max_width - 35)
            if lines:
                first, *rest = lines
                draw_lines([bullet_prefix + first], body_font)
                if rest:
                    draw_lines(rest, body_font, indent=28)
            y += 4
            continue

        if block_type == "numbered":
            parts = value.split(". ", 1)
            prefix = f"{parts[0]}. "
            body = parts[1] if len(parts) > 1 else value
            prefix_width = draw.textbbox((0, 0), prefix, font=bold_body_font)[2]
            lines = wrap_text(draw, body, body_font, max_width - prefix_width - 10)
            if lines:
                first, *rest = lines
                ensure_space(40)
                draw.text((MARGIN_X, y), prefix, font=bold_body_font, fill=TEXT_COLOR)
                draw.text((MARGIN_X + prefix_width, y), first, font=body_font, fill=TEXT_COLOR)
                line_height = draw.textbbox((0, 0), "Ag", font=body_font)[3] + LINE_SPACING
                y += line_height
                for line in rest:
                    draw.text((MARGIN_X + prefix_width, y), line, font=body_font, fill=TEXT_COLOR)
                    y += line_height
            y += 4
            continue

        if block_type == "code":
            lines = wrap_text(draw, value.replace("\t", "    "), code_font, max_width - 30)
            bbox = draw.textbbox((0, 0), "Ag", font=code_font)
            line_height = (bbox[3] - bbox[1]) + LINE_SPACING
            block_height = max(line_height * max(len(lines), 1) + 18, line_height + 18)
            ensure_space(block_height)
            draw.rounded_rectangle(
                (MARGIN_X, y, PAGE_SIZE[0] - MARGIN_X, y + block_height),
                radius=12,
                fill=(245, 245, 245),
                outline=(220, 220, 220),
            )
            inner_y = y + 10
            for line in lines:
                draw.text((MARGIN_X + 15, inner_y), line, font=code_font, fill=TEXT_COLOR)
                inner_y += line_height
            y += block_height + 8
            continue

        lines = wrap_text(draw, value, body_font, max_width)
        draw_lines(lines, body_font)
        y += 4

    pages.append(page)

    rgb_pages = [img.convert("RGB") for img in pages]
    first, rest = rgb_pages[0], rgb_pages[1:]
    first.save(destination, save_all=True, append_images=rest, resolution=150.0)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Export a Markdown document to a simple PDF.")
    parser.add_argument("source", type=Path, help="Markdown file to export")
    parser.add_argument(
        "--output",
        type=Path,
        help="Destination PDF path. Defaults to the same name with .pdf extension.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    source = args.source
    output = args.output or source.with_suffix(".pdf")
    render_markdown_to_pdf(source, output)
    print(output)


if __name__ == "__main__":
    main()
