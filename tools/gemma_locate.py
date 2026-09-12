"""Locate neckerchief fastenings and save PNG crops under crops/NC.

Requires Python 3.14 and Pillow. Processes this directory's PNG/JPEG/WebP files.
Each request contains one image; existing results resume by filename and SHA-256.
Normal resumes silently skip unchanged images whose crop step is complete.
Missing crops are restored from existing locations; --crop-only rebuilds crops.
Directory scanning and SHA-256 checks still run to find new or changed images.
Image jobs run concurrently (default: 4, configurable with --workers).
The main thread saves completed results and crops in completion order.
Coordinates use the EXIF-upright image, in pixels, as [left, top, right, bottom].
NC.status is visible, estimated (occluded), or unavailable. NC.box encloses the
visible fastening or its inferred position; NC.crop is the region for review.
Both coordinates are null when status is unavailable. Visible crops retain the
minimum --crop-size rule; estimated crops use --crop-size (default: 256) without
enlargement, limited by the source image dimensions. NC.crop_file records the
crop's relative path, or null. --crop-only uses locations.json without a model.
Original images are unchanged. No clothing is graded.
Uses gemma_batch for image encoding and structured model requests.
Thinking defaults to off (128 output tokens); --think enables it (4096 tokens).
--max-tokens overrides that budget. Changing inference settings requires --redo
when locations.json already exists.
"""

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import time

from PIL import ExifTags, Image, ImageOps

from gemma_batch import MODEL, SERVER, call_gemma, encode_image


PROMPT = """Locate the small central fastening that gathers the neckerchief at the front of the chest. It may be a ring, a slider, or a fabric knot, regardless of its color or shape.
If the whole fastening is visible, use status=visible and return one tight bounding box enclosing the fastening itself, excluding the long hanging tails, collar, and face.
If the fastening is partly or completely obscured by hands, hair, or another object, infer its expected position from the collar opening, neck base, chest orientation, and visible neckerchief fabric. Use status=estimated and return a tight box around the expected fastening itself. Center the box on where the fastening should be behind the obstruction; do not box the whole hand, hair, or other covering object. Occlusion alone is not a reason to return unavailable.
Use status=unavailable and bbox=null only when there is insufficient visual context to infer a reasonable fastening position inside the image, including when the target area is outside the frame or there is no visible neckerchief or supporting garment structure.
Use image coordinates normalized to 0..1000. The origin is the top-left of the full image. bbox must be [left, top, right, bottom], where horizontal coordinates increase to the right and vertical coordinates increase downward.
Return only a JSON object with status and bbox, using integer coordinates and no explanation. Example: {"status":"estimated","bbox":[left,top,right,bottom]}"""

SCHEMA = {"anyOf": [
    {"type": "object", "properties": {
        "status": {"type": "string", "enum": ["visible", "estimated"]},
        "bbox": {"type": "array", "items": {"type": "integer", "minimum": 0, "maximum": 1000},
                 "minItems": 4, "maxItems": 4},
    }, "required": ["status", "bbox"], "additionalProperties": False},
    {"type": "object", "properties": {
        "status": {"const": "unavailable"}, "bbox": {"type": "null"},
    }, "required": ["status", "bbox"], "additionalProperties": False},
]}


def locate(path, settings):
    with Image.open(path) as original:
        width, height = original.size
        if original.getexif().get(ExifTags.Base.Orientation) in (5, 6, 7, 8): width, height = height, width
    data_url = encode_image(path)
    think = settings["reasoning_effort"] != "none"
    started = time.perf_counter()
    result = call_gemma(
        [data_url], prompt=settings["prompt"], schema=settings["schema"],
        think=think, model=settings["model"], server=settings["server"],
        max_tokens=settings["max_tokens"], timeout=600 if think else 180,
    )
    seconds = time.perf_counter() - started
    normalized = result["bbox"]
    box = crop = None
    if normalized is not None:
        left, top, right, bottom = normalized
        box = [round(left * width / 1000), round(top * height / 1000),
               round(right * width / 1000), round(bottom * height / 1000)]
        center_x, center_y = (box[0] + box[2]) / 2, (box[1] + box[3]) / 2
        side = settings["crop_size"]
        if result["status"] == "visible": side = max(side, box[2] - box[0], box[3] - box[1])
        crop_width, crop_height = min(width, side), min(height, side)
        x = max(0, min(width - crop_width, round(center_x - crop_width / 2)))
        y = max(0, min(height - crop_height, round(center_y - crop_height / 2)))
        crop = [x, y, x + crop_width, y + crop_height]
    return {"size": [width, height], "parts": {"NC": {"status": result["status"], "box": box, "crop": crop}},
            "request_seconds": seconds}


def process_image(path, previous, settings, crop_only):
    sha256 = hashlib.sha256(path.read_bytes()).hexdigest()
    if crop_only:
        if previous is None: return None, False
        if previous["sha256"] != sha256:
            raise RuntimeError(f"原图内容已改变，请先重新定位：{path.name}")
    if previous is not None and previous["sha256"] == sha256: return previous, False
    return {"sha256": sha256, **locate(path, settings)}, True


def save(index, path):
    temporary = path.with_suffix(".json.tmp")
    temporary.write_text(json.dumps(index, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    temporary.replace(path)


def crop_image(root, source, box):
    directory = root / "crops" / "NC"
    # Retain a non-PNG suffix so a.jpg and a.png produce different crop names.
    name = source.name if source.suffix.lower() == ".png" else source.name + ".png"
    destination = directory / name
    if box is None:
        destination.unlink(missing_ok=True)
        return None
    with Image.open(source) as original:
        image = ImageOps.exif_transpose(original).convert("RGB").crop(tuple(box))
    directory.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix(".png.tmp")
    image.save(temporary, format="PNG")
    temporary.replace(destination)
    return destination.relative_to(root).as_posix()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("directory", type=Path, help="Folder containing images (not recursive)")
    parser.add_argument("--model", default=MODEL)
    parser.add_argument("--server", default=SERVER)
    parser.add_argument("--workers", type=int, default=4, help="Concurrent image jobs (default: 4)")
    parser.add_argument("--think", action=argparse.BooleanOptionalAction, default=False, help="Enable thinking (default: off)")
    parser.add_argument("--max-tokens", type=int, help="Output budget including thinking (default: 128 off, 4096 on)")
    parser.add_argument("--crop-size", type=int, default=256, help="Crop side in upright pixels: minimum for visible, fixed for estimated (default: 256)")
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--redo", action="store_true", help="Recompute all locations instead of resuming")
    mode.add_argument("--crop-only", action="store_true", help="Crop from locations.json without calling a model")
    args = parser.parse_args()
    root = args.directory.resolve()
    if not (root / "crops" / "NC").resolve().is_relative_to(root):
        raise RuntimeError("裁剪目录不在输入目录内。")
    destination = root / "locations.json"
    max_tokens = args.max_tokens if args.max_tokens is not None else (4096 if args.think else 128)
    settings = {"model": args.model, "server": args.server.rstrip("/"), "part": "NC",
                "prompt": PROMPT, "schema": SCHEMA, "reasoning_effort": "high" if args.think else "none",
                "temperature": 0, "max_tokens": max_tokens, "crop_size": args.crop_size}
    previous_index = None
    if destination.exists() or args.crop_only:
        previous_index = json.loads(destination.read_text(encoding="utf-8"))
    if previous_index is not None and not args.redo:
        index = previous_index
        if index["settings"]["schema"] != SCHEMA or (not args.crop_only and index["settings"] != settings):
            raise RuntimeError("定位设置已改变；请使用 --redo 重新定位，不能混用已有结果。")
    else:
        index = {"created_at": datetime.now(timezone.utc).isoformat(), "settings": settings,
                 "coordinates": "exif_upright_pixels_xyxy", "images": {}}
    files = sorted((p for p in root.iterdir() if p.is_file() and p.suffix.lower() in
                    (".png", ".jpg", ".jpeg", ".webp")), key=lambda p: p.name.casefold())
    names = {p.name for p in files}
    if previous_index is not None:
        for name in previous_index["images"].keys() - names:
            crop_image(root, root / name, None)
    save_needed = previous_index is None or args.redo or bool(index["images"].keys() - names)
    index["images"] = {name: row for name, row in index["images"].items() if name in names}
    started = time.perf_counter()
    reused = calls = skipped = 0
    operation = ("Crop only (no model requests)" if args.crop_only else
                 f"Model: {args.model} | thinking: {'on' if args.think else 'off'}")
    print(f"Folder: {root}\n{operation} | workers: {args.workers} | part: NC | images: {len(files)}", flush=True)
    pool = ThreadPoolExecutor(max_workers=args.workers)
    try:
        pending = {pool.submit(process_image, path, index["images"].get(path.name), settings, args.crop_only): path
                   for path in sorted(files, key=lambda path: path.name in index["images"])}
        for future in as_completed(pending):
            path = pending[future]
            row, is_new = future.result()
            if row is None:
                print(f"no location record: {path.name}", flush=True)
                continue
            part = row["parts"]["NC"]
            if not is_new and not args.crop_only and "crop_file" in part and (
                part["crop_file"] is None or (root / part["crop_file"]).is_file()
            ):
                skipped += 1
                continue
            if is_new:
                index["images"][path.name] = row
                save(index, destination)
                calls += 1
                print(f"[{calls + reused}] {part['status']} {path.name} | {row['request_seconds']:.2f}s", flush=True)
            else:
                reused += 1
                print(f"[{calls + reused}] restoring crop: {path.name}", flush=True)
            part["crop_file"] = crop_image(root, path, part["crop"])
            save(index, destination)
            save_needed = False
            print(f"  crop: {part['crop_file'] if part['crop_file'] is not None else 'unavailable'}", flush=True)
    finally:
        pool.shutdown(wait=True, cancel_futures=True)
    if save_needed: save(index, destination)
    for directory in (root / "crops" / "NC", root / "crops"):
        if directory.is_dir() and not any(directory.iterdir()): directory.rmdir()
    counts = {"visible": 0, "estimated": 0, "unavailable": 0}
    for row in index["images"].values(): counts[row["parts"]["NC"]["status"]] += 1
    located = len(index["images"])
    print(f"Done: visible {counts['visible']}, estimated {counts['estimated']}, unavailable {counts['unavailable']}, no location record {len(files) - located}, new calls {calls}, reused {reused}, skipped {skipped}.")
    print(f"Elapsed: {time.perf_counter() - started:.2f}s\nIndex: {destination}")


if __name__ == "__main__":
    main()
