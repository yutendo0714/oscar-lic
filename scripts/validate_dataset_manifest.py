#!/usr/bin/env python3
from __future__ import annotations

import argparse
from hashlib import sha256
import json
from pathlib import Path

from PIL import Image


def digest(path: Path) -> str:
    h = sha256()
    with path.open("rb") as f:
        while chunk := f.read(1024 * 1024):
            h.update(chunk)
    return h.hexdigest()


def main() -> int:
    parser = argparse.ArgumentParser(description="Validate paths, hashes and dimensions in an image JSONL manifest")
    parser.add_argument("manifest", type=Path)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--require-hash", action="store_true")
    args = parser.parse_args()
    errors=[]; count=0; seen=set()
    for line_no,line in enumerate(args.manifest.read_text(encoding='utf-8').splitlines(),1):
        if not line.strip(): continue
        count += 1
        try: row=json.loads(line)
        except Exception as e:
            errors.append(f"line {line_no}: invalid JSON: {e}"); continue
        sid=row.get('sample_id') or row.get('relative_path')
        if sid in seen: errors.append(f"line {line_no}: duplicate sample {sid}")
        seen.add(sid)
        rel=row.get('relative_path')
        if not rel: errors.append(f"line {line_no}: missing relative_path"); continue
        path=(args.root/rel).resolve()
        try: path.relative_to(args.root.resolve())
        except ValueError: errors.append(f"line {line_no}: path escapes root"); continue
        if not path.is_file(): errors.append(f"line {line_no}: file missing: {rel}"); continue
        expected=row.get('sha256')
        if args.require_hash and not expected: errors.append(f"line {line_no}: SHA256 required")
        if expected and digest(path) != expected: errors.append(f"line {line_no}: SHA256 mismatch: {rel}")
        if row.get('width') and row.get('height'):
            try:
                with Image.open(path) as im: size=im.size
                if size != (row['width'],row['height']): errors.append(f"line {line_no}: dimension mismatch {size}")
            except Exception as e: errors.append(f"line {line_no}: image read failed: {e}")
    for e in errors: print('ERROR:',e)
    if errors: return 1
    print(f"OK: {count} records validated")
    return 0

if __name__=='__main__': raise SystemExit(main())
