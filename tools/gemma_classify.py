"""Keep images matching a configurable prompt and permanently delete nonmatches.

Requires Python 3.14 and Pillow. Uses gemma_batch for independent single-image
requests with no reference image. Edit PROMPT below or pass --prompt to change
the requirement. The structured answer is {"matches": true} or {"matches": false}.

Only images directly inside the input directory are processed. Matches stay
in place; nonmatches are deleted immediately. No folders, database, or result
files are created. Rerunning evaluates all remaining top-level images again.
Four requests run concurrently by default. The main thread deletes each
nonmatch when its result arrives. Request or parsing errors stop the run;
the affected image is left in place.

CLI:
    python tools/gemma_classify.py "IMAGE_DIR" [--prompt "REQUIREMENT"] [--no-think] [--workers 4]
    Thinking is enabled by default; --no-think disables it.
"""

import argparse
from contextlib import closing
from pathlib import Path

from gemma_batch import MODEL, SERVER, run_directory


PROMPT = "判断图中人物是否穿着深蓝色水手服。"
SCHEMA = {
    "type": "object",
    "properties": {
        "matches": {"type": "boolean", "description": "符合要求为 true，不符合要求为 false"},
    },
    "required": ["matches"],
    "additionalProperties": False,
}


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("directory", type=Path, help="Candidate image directory (not recursive)")
    parser.add_argument("--prompt", default=PROMPT, help="Requirement to check (default: PROMPT in this script)")
    parser.add_argument("--think", action=argparse.BooleanOptionalAction, default=True, help="Enable thinking (default: on)")
    parser.add_argument("--server", default=SERVER)
    parser.add_argument("--model", default=MODEL)
    parser.add_argument("--max-tokens", type=int, help="Output budget including thinking (default: 1024 off, 4096 on)")
    parser.add_argument("--timeout", type=float, default=600, help="Request timeout in seconds")
    parser.add_argument("--workers", type=int, default=4, help="Concurrent requests (default: 4)")
    args = parser.parse_args()
    root = args.directory.resolve()
    max_tokens = args.max_tokens if args.max_tokens is not None else (4096 if args.think else 1024)
    rows = run_directory(
        root, prompt=args.prompt, schema=SCHEMA,
        think=args.think, server=args.server, model=args.model,
        max_tokens=max_tokens, timeout=args.timeout, workers=args.workers,
    )
    counts = {"KEEP": 0, "DELETE": 0}
    print(f"Folder: {root}\nPrompt: {args.prompt}\nModel: {args.model} | think: {'on' if args.think else 'off'} | workers: {args.workers}", flush=True)
    with closing(rows):
        for number, row in enumerate(rows, start=1):
            source = Path(row["image"])
            result = "KEEP" if row["result"]["matches"] else "DELETE"
            if result == "DELETE": source.unlink()
            counts[result] += 1
            print(f"[{number}] {source.name} -> {result} | {row['seconds']:.3f}s", flush=True)
    print(f"Done: kept {counts['KEEP']}, deleted {counts['DELETE']}.")


if __name__ == "__main__":
    main()
