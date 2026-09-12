"""Run a configurable Gemma task on each image directly inside a directory.

Requires Python 3.14 and Pillow. Edit the task JSON's prompt and schema to
change the task. Each request has its own context and contains one candidate.
An optional reference is sent first, followed by the candidate. Originals are
unchanged; subdirectories are ignored.

CLI:
    python tools/gemma_batch.py "IMAGE_DIR" --task "TASK.json" \
        [--think] [--workers 4] [--output results.jsonl]
    python tools/gemma_batch.py "IMAGE_DIR" --reference "REFERENCE.png" \
        --task tools/gemma_compare.task.json

The output is overwritten on each run and flushed after every result.
Each JSONL line contains image/reference paths, seconds, and result (the JSON
matching your schema). Reference is null when omitted. The default output is
IMAGE_DIR/gemma_results.jsonl.
The default concurrency is 1. With multiple workers, results arrive in
completion order. Use contextlib.closing() when consuming only part of a run.

Reuse from Python:
    for row in run_directory(directory, prompt=prompt, schema=schema):
        print(row["result"])

For other workflows, encode_image() prepares a data URL and call_gemma()
accepts any ordered sequence of image data URLs, returning just the task JSON.
"""

import argparse
import base64
from collections.abc import Iterator, Sequence
from concurrent.futures import ThreadPoolExecutor, as_completed
from contextlib import closing
from io import BytesIO
import json
from pathlib import Path
import time
from urllib.request import Request, urlopen

from PIL import Image, ImageOps


SERVER = "http://127.0.0.1:1234"
MODEL = "gemma3"


def encode_image(path: Path) -> str:
    """Encode an upright image at its original resolution without changing the file."""
    with Image.open(path) as original:
        image = ImageOps.exif_transpose(original).convert("RGB")
        buffer = BytesIO()
        image.save(buffer, format="PNG")
    return "data:image/png;base64," + base64.b64encode(buffer.getvalue()).decode("ascii")


def call_gemma(
    image_urls: Sequence[str], *, prompt: str, schema: dict,
    think: bool = False, server: str = SERVER, model: str = MODEL,
    max_tokens: int = 4096, timeout: float = 600,
) -> dict:
    """Send one independent multi-image request and return the structured answer."""
    content = []
    for number, url in enumerate(image_urls, start=1):
        if len(image_urls) > 1: content.append({"type": "text", "text": f"Image {number}:"})
        content.append({"type": "image_url", "image_url": {"url": url}})
    content.append({"type": "text", "text": prompt})
    payload = {
        "model": model,
        "messages": [{"role": "user", "content": content}],
        "temperature": 0,
        "max_tokens": max_tokens,
        # LM Studio's Gemma uses none = off; high enables thinking.
        "reasoning_effort": "high" if think else "none",
        "stream": False,
        "response_format": {
            "type": "json_schema",
            "json_schema": {"name": "task_result", "strict": True, "schema": schema},
        },
    }
    request = Request(
        server.rstrip("/") + "/v1/chat/completions",
        data=json.dumps(payload, ensure_ascii=False).encode("utf-8"),
        headers={"Content-Type": "application/json"},
    )
    with urlopen(request, timeout=timeout) as response:
        reply = json.load(response)
    return json.loads(reply["choices"][0]["message"]["content"])


def run_directory(
    directory: Path, *, prompt: str, schema: dict, reference: Path | None = None,
    think: bool = False, server: str = SERVER, model: str = MODEL,
    max_tokens: int = 4096, timeout: float = 600, workers: int = 1,
) -> Iterator[dict]:
    """Yield completed tasks, sharing the optional encoded reference across workers."""
    directory = Path(directory).resolve()
    reference = Path(reference).resolve() if reference is not None else None
    extensions = Image.registered_extensions()
    images = sorted(
        (path.resolve() for path in directory.iterdir()
         if path.is_file() and path.suffix.lower() in extensions and path.resolve() != reference),
        key=lambda path: path.name.casefold(),
    )
    reference_urls = [encode_image(reference)] if reference is not None else []

    def process(path: Path) -> dict:
        image_url = encode_image(path)
        started = time.perf_counter()
        result = call_gemma(
            [*reference_urls, image_url], prompt=prompt, schema=schema,
            think=think, server=server, model=model, max_tokens=max_tokens, timeout=timeout,
        )
        return {"image": str(path), "reference": str(reference) if reference is not None else None,
                "seconds": round(time.perf_counter() - started, 3), "result": result}

    pool = ThreadPoolExecutor(max_workers=workers)
    try:
        futures = [pool.submit(process, path) for path in images]
        for future in as_completed(futures): yield future.result()
    finally:
        pool.shutdown(wait=True, cancel_futures=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("directory", type=Path, help="Candidate image directory (not recursive)")
    parser.add_argument("--reference", type=Path, help="Optional reference image; sent before each candidate")
    parser.add_argument("--task", type=Path, required=True, help="JSON file containing prompt and schema")
    parser.add_argument("--think", action=argparse.BooleanOptionalAction, default=False, help="Enable thinking (default: off)")
    parser.add_argument("--server", default=SERVER)
    parser.add_argument("--model", default=MODEL)
    parser.add_argument("--max-tokens", type=int, default=4096, help="Output budget including thinking tokens")
    parser.add_argument("--timeout", type=float, default=600, help="Request timeout in seconds")
    parser.add_argument("--workers", type=int, default=1, help="Concurrent requests (default: 1)")
    parser.add_argument("--output", type=Path, help="JSONL output path (overwritten)")
    args = parser.parse_args()
    task = json.loads(args.task.read_text(encoding="utf-8"))
    destination = args.output if args.output is not None else args.directory / "gemma_results.jsonl"
    rows = run_directory(
        args.directory, reference=args.reference, prompt=task["prompt"], schema=task["schema"],
        think=args.think, server=args.server, model=args.model,
        max_tokens=args.max_tokens, timeout=args.timeout, workers=args.workers,
    )
    print(f"Model: {args.model} | think: {'on' if args.think else 'off'} | workers: {args.workers}", flush=True)
    with closing(rows), destination.open("w", encoding="utf-8") as output:
        for number, row in enumerate(rows, start=1):
            output.write(json.dumps(row, ensure_ascii=False) + "\n")
            output.flush()
            print(f"[{number}] {Path(row['image']).name} | {row['seconds']:.3f}s | "
                  + json.dumps(row["result"], ensure_ascii=False), flush=True)
    print(f"Saved: {destination.resolve()}")


if __name__ == "__main__":
    main()
