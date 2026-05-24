import os
import re
import json
import base64
import tempfile
import argparse

from dotenv import load_dotenv
from pdf2image import convert_from_path
from PIL import Image, ImageDraw, ImageFont
from langchain_core.messages import HumanMessage
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_openai import ChatOpenAI

load_dotenv()

MODEL_NAME = "gemini-3.1-pro-preview"

# ---------------------------------------------------------------------------
# Image utilities
# ---------------------------------------------------------------------------

_TEMP_DIR = tempfile.gettempdir()


def load_and_resize(path: str, max_px: int = 1024) -> tuple:
    img = Image.open(path).convert("RGB")
    orig_w, orig_h = img.size
    scale = min(max_px / max(orig_w, orig_h), 1.0)
    if scale < 1.0:
        new_w = int(orig_w * scale)
        new_h = int(orig_h * scale)
        img_resized = img.resize((new_w, new_h), Image.LANCZOS)
    else:
        img_resized = img
    temp_path = os.path.join(_TEMP_DIR, "flyer_llm_input.jpg")
    img_resized.save(temp_path, "JPEG", quality=85)
    return temp_path, orig_w, orig_h


def draw_annotations(image_path: str, promotions: list) -> Image.Image:
    img = Image.open(image_path).convert("RGBA")
    overlay = Image.new("RGBA", img.size, (0, 0, 0, 0))
    draw_overlay = ImageDraw.Draw(overlay)
    for promo in promotions:
        x1, y1, x2, y2 = promo["bbox"]
        color = tuple(promo["color"])
        fill = color if len(color) == 4 else (*color[:3], 80)
        draw_overlay.rectangle([x1, y1, x2, y2], fill=fill)
    img = Image.alpha_composite(img, overlay)
    draw = ImageDraw.Draw(img)
    try:
        label_font = ImageFont.truetype("arial.ttf", 28)
        circle_font = ImageFont.truetype("arial.ttf", 32)
    except OSError:
        label_font = ImageFont.load_default()
        circle_font = ImageFont.load_default()
    for idx, promo in enumerate(promotions, 1):
        x1, y1, x2, y2 = promo["bbox"]
        outline = tuple(promo["outline"])[:3]
        for t in range(14):
            if x1 + t >= x2 - t or y1 + t >= y2 - t:
                break
            draw.rectangle([x1 + t, y1 + t, x2 - t, y2 - t], outline=outline)
        lines = promo["label"].split("\\n") if "\\n" in promo["label"] else promo["label"].split("\n")
        padding = 8
        line_height = 36
        panel_h = padding * 2 + len(lines) * line_height
        panel_w = max((len(line) * 17 + padding * 2) for line in lines)
        panel_w = min(panel_w, x2 - x1)
        draw.rectangle([x1, y1, x1 + panel_w, y1 + panel_h], fill=(*outline, 220))
        for i, line in enumerate(lines):
            draw.text(
                (x1 + padding, y1 + padding + i * line_height),
                line,
                fill=(255, 255, 255),
                font=label_font,
            )
        circle_r = 28
        cx = x2 - circle_r - 10
        cy = y1 + circle_r + 10
        draw.ellipse(
            [cx - circle_r, cy - circle_r, cx + circle_r, cy + circle_r],
            fill=(*outline, 255),
        )
        idx_str = str(idx)
        bbox_text = draw.textbbox((0, 0), idx_str, font=circle_font)
        tw = bbox_text[2] - bbox_text[0]
        th = bbox_text[3] - bbox_text[1]
        draw.text((cx - tw // 2, cy - th // 2), idx_str, fill=(255, 255, 255), font=circle_font)
    return img


def save_jpeg(img: Image.Image, path: str, quality: int = 92) -> None:
    rgb = img.convert("RGB") if img.mode in ("RGBA", "P") else img
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    rgb.save(path, "JPEG", quality=quality)


# ---------------------------------------------------------------------------
# Color / validation utilities
# ---------------------------------------------------------------------------

_DEFAULT_COLORS = [
    ((220, 50, 50, 80), (180, 30, 30)),
    ((50, 100, 220, 80), (30, 70, 180)),
    ((50, 180, 80, 80), (30, 140, 60)),
    ((200, 140, 50, 80), (160, 100, 30)),
    ((140, 50, 200, 80), (100, 30, 160)),
    ((50, 180, 200, 80), (30, 140, 160)),
    ((200, 80, 140, 80), (160, 50, 100)),
    ((100, 200, 80, 80), (70, 160, 50)),
    ((200, 200, 50, 80), (160, 160, 30)),
    ((100, 50, 200, 80), (70, 30, 160)),
    ((50, 200, 200, 80), (30, 160, 160)),
    ((200, 120, 50, 80), (160, 90, 30)),
]


def assign_colors(promotions: list) -> list:
    for i, promo in enumerate(promotions):
        if not promo.get("color"):
            fill, outline = _DEFAULT_COLORS[i % len(_DEFAULT_COLORS)]
            promo["color"] = fill
            promo["outline"] = outline
    return promotions


# ---------------------------------------------------------------------------
# JSON parsing utilities
# ---------------------------------------------------------------------------

def _extract_json_array(raw: str) -> list:
    cleaned = re.sub(r"```(?:json)?\s*\n?", "", raw)
    cleaned = cleaned.replace("```", "").strip()
    try:
        data = json.loads(cleaned)
        if isinstance(data, list):
            return data
    except json.JSONDecodeError:
        pass
    match = re.search(r"\[[\s\S]*\]", cleaned)
    if match:
        return json.loads(match.group())
    raise ValueError(f"No valid JSON array found in LLM response:\n{raw[:400]}")


def _process_promotion_item(item: dict, img_w: int, img_h: int) -> dict:
    bbox_raw = item["bbox"]
    if all(isinstance(v, float) and 0.0 <= v <= 1.0 for v in bbox_raw):
        x1 = int(bbox_raw[0] * img_w)
        y1 = int(bbox_raw[1] * img_h)
        x2 = int(bbox_raw[2] * img_w)
        y2 = int(bbox_raw[3] * img_h)
    else:
        x1, y1, x2, y2 = (int(v) for v in bbox_raw)
    x1, x2 = min(x1, x2), max(x1, x2)
    y1, y2 = min(y1, y2), max(y1, y2)
    # print(f"Raw bbox: {bbox_raw} → Scaled bbox: {[x1, y1, x2, y2]}")
    raw_color = item.get("color", [255, 100, 100, 80])
    raw_outline = item.get("outline", [200, 50, 50])
    color = tuple(int(c) for c in raw_color[:4]) if len(raw_color) >= 4 else (*[int(c) for c in raw_color[:3]], 80)
    outline = tuple(int(c) for c in raw_outline[:3])
    return {
        "label": str(item["label"]),
        "bbox": [x1, y1, x2, y2],
        "color": color,
        "outline": outline,
    }


def parse_promotions_json(raw: str, img_w: int, img_h: int) -> list:
    data = _extract_json_array(raw)
    return [_process_promotion_item(item, img_w, img_h) for item in data]


# ---------------------------------------------------------------------------
# LLM call
# ---------------------------------------------------------------------------

def process_image(image_path: str, prompt: str, model_name: str = "gpt-4.1-mini") -> str:
    if "gemini" in model_name:
        model = ChatGoogleGenerativeAI(model=model_name)
    else:
        model = ChatOpenAI(model=model_name)
    image_bytes = open(image_path, "rb").read()
    image_base64 = base64.b64encode(image_bytes).decode("utf-8")
    message = HumanMessage(
        content=[
            {"type": "text", "text": prompt},
            {"type": "image", "base64": image_base64, "mime_type": "image/jpeg"},
        ]
    )
    response = model.invoke([message])
    print(response.content)
    print(response.usage_metadata)
    return response.content


# ---------------------------------------------------------------------------
# Detection prompt
# ---------------------------------------------------------------------------

_DETECTION_PROMPT_TEMPLATE = """You are a grocery flyer annotation expert.

Layout context:
{layout}

Detect ALL distinct promotional offers in this grocery flyer and return bounding box coordinates.
{annotation}

Return a JSON array only — no prose, no markdown fences. Each element must be:
{{
  "label": "price + product name(s) (use \\n for line breaks, e.g. \\"$3.99\\nStrawberries\\")",
  "bbox": [x1, y1, x2, y2],   // decimal fractions 0.0–1.0 of image width/height
  "color": [R, G, B, 80],     // semi-transparent fill
  "outline": [R, G, B]        // darker border
}}

Rules:
- One box per unique price offer
- Box must cover the price text AND all associated product images/names
- Even if the box overlaps with other boxes, it should still cover the entire offer (price + products)
- Prefer tight boxes with minimal whitespace
- Use a distinct color per promotion"""


# ---------------------------------------------------------------------------
# Pipeline
# ---------------------------------------------------------------------------

def pdf_to_images(pdf_path: str, images_dir: str, start_page: int = 1, end_page: int | None = None) -> list[str]:
    os.makedirs(images_dir, exist_ok=True)
    pdf_stem = os.path.splitext(os.path.basename(pdf_path))[0]
    pages = convert_from_path(pdf_path, dpi=150, first_page=start_page, last_page=end_page)
    saved = []
    for i, page in enumerate(pages):
        page_num = start_page + i
        out_path = os.path.join(images_dir, f"{pdf_stem}_{page_num}.jpg")
        if os.path.exists(out_path):
            print(f"Skipping page {page_num} (already exists) → {out_path}")
        else:
            page.save(out_path, "JPEG", quality=90)
            print(f"Saved page {page_num} → {out_path}")
        saved.append(out_path)
    return saved


def annotate_image(image_path: str, output_dir: str, annotation_dir: str) -> str:
    stem = os.path.splitext(os.path.basename(image_path))[0]
    out_path = os.path.join(output_dir, f"{stem}.jpg")
    if os.path.exists(out_path):
        print(f"  Skipping (already annotated) → {out_path}")
        return out_path

    resized_path, orig_w, orig_h = load_and_resize(image_path, max_px=1024)

    prompt = _DETECTION_PROMPT_TEMPLATE.format(layout="Not available", annotation="")
    raw = process_image(resized_path, prompt, model_name=MODEL_NAME)
    raw = raw[0]["text"].replace("```json", "").replace("```", "").strip()

    json_path = os.path.join(annotation_dir, f"{stem}.json")
    with open(json_path, "w") as f:
        f.write(raw)
    print(f"  Annotation → {json_path}")

    promotions = parse_promotions_json(raw, orig_w, orig_h)
    promotions = assign_colors(promotions)
    print(f"  Detected {len(promotions)} promotions")

    annotated = draw_annotations(image_path, promotions)

    save_jpeg(annotated, out_path)
    print(f"  Saved → {out_path}")
    return out_path


def main():
    parser = argparse.ArgumentParser(description="Annotate promotions in a PDF flyer.")
    parser.add_argument("--pdf", required=True, help="Path to input PDF file")
    parser.add_argument("--output", required=True, help="Folder for annotated output images")
    parser.add_argument("--images_dir", default="cropped_images", help="Folder for PDF page images (default: cropped_images)")
    parser.add_argument("--annotation", required=True, help="Folder for JSON annotation responses")
    parser.add_argument("--start_page", type=int, default=1, help="First page to process (1-indexed, default: 1)")
    parser.add_argument("--end_page", type=int, default=None, help="Last page to process inclusive (default: all pages)")
    args = parser.parse_args()

    print(f"Converting PDF: {args.pdf} (pages {args.start_page}–{args.end_page or 'end'})")
    image_paths = pdf_to_images(args.pdf, args.images_dir, start_page=args.start_page, end_page=args.end_page)
    print(f"Converted {len(image_paths)} pages\n")

    os.makedirs(args.output, exist_ok=True)
    os.makedirs(args.annotation, exist_ok=True)

    for i, image_path in enumerate(image_paths, 1):
        print(f"[{i}/{len(image_paths)}] Annotating: {image_path}")
        annotate_image(image_path, args.output, args.annotation)

    print(f"\nDone. Annotated images saved to: {args.output}")


if __name__ == "__main__":
    main()
