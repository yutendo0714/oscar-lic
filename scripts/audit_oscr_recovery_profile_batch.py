#!/usr/bin/env python3
"""Batch audit OSCR section-level recovery behavior over stream directories."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from oscarlic.bitstream import BitstreamError, parse_container, parse_container_recovery  # noqa: E402


SECTION_BASE_MAIN = 2
SECTION_TEXT_MAIN = 12


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def section_by_type(parsed, section_type: int):
    matches = [section for section in parsed.sections if section.section_type == section_type]
    if len(matches) != 1:
        return None
    return matches[0]


def corrupt_one_byte(data: bytes, offset: int) -> bytes:
    buf = bytearray(data)
    buf[offset] ^= 0x01
    return bytes(buf)


def try_default_parse(data: bytes) -> dict:
    try:
        parsed = parse_container(data, verify_crc=True)
        return {"ok": True, "section_count": len(parsed.sections), "bpp": parsed.bpp}
    except Exception as exc:  # noqa: BLE001 - manifest records exact failure modes
        return {"ok": False, "error_type": type(exc).__name__, "error": str(exc)}


def try_recovery(data: bytes) -> dict:
    try:
        recovered = parse_container_recovery(data)
        return {
            "ok": True,
            "section_count": len(recovered.sections),
            "rejected_section_count": len(recovered.rejected_sections),
            "rejected_section_types": [section.section_type for section in recovered.rejected_sections],
            "file_crc_ok": recovered.file_crc_ok,
        }
    except Exception as exc:  # noqa: BLE001 - manifest records exact failure modes
        return {"ok": False, "error_type": type(exc).__name__, "error": str(exc)}


def audit_stream(path: Path) -> dict:
    data = path.read_bytes()
    parsed = parse_container(data, verify_crc=True)
    base = section_by_type(parsed, SECTION_BASE_MAIN)
    text_main = section_by_type(parsed, SECTION_TEXT_MAIN)
    if base is None:
        raise BitstreamError(f"{path} has no unique BASE_MAIN section")

    row = {
        "stream": str(path),
        "stream_sha256": sha256_bytes(data),
        "total_bytes": parsed.total_bytes,
        "bpp": parsed.bpp,
        "section_types": [section.section_type for section in parsed.sections],
        "base_payload_sha256": sha256_bytes(base.payload),
        "has_text_main": text_main is not None,
    }

    base_flip = corrupt_one_byte(data, base.offset)
    row["corrupt_base_main_first_byte"] = {
        "default_parse": try_default_parse(base_flip),
        "recovery_parse": try_recovery(base_flip),
    }

    if text_main is not None:
        text_flip = corrupt_one_byte(data, text_main.offset)
        text_recovery = try_recovery(text_flip)
        recovered_base_sha = None
        if text_recovery["ok"]:
            recovered = parse_container_recovery(text_flip)
            recovered_base = section_by_type(recovered, SECTION_BASE_MAIN)
            if recovered_base is not None:
                recovered_base_sha = sha256_bytes(recovered_base.payload)
        row["corrupt_text_main_first_byte"] = {
            "default_parse": try_default_parse(text_flip),
            "recovery_parse": text_recovery,
            "base_payload_sha256_unchanged": recovered_base_sha == row["base_payload_sha256"],
        }

    return row


def summarize(rows: list[dict]) -> dict:
    text_rows = [row for row in rows if row["has_text_main"]]
    text_recovered = [
        row for row in text_rows
        if row["corrupt_text_main_first_byte"]["recovery_parse"]["ok"]
        and row["corrupt_text_main_first_byte"]["base_payload_sha256_unchanged"]
        and SECTION_TEXT_MAIN in row["corrupt_text_main_first_byte"]["recovery_parse"]["rejected_section_types"]
    ]
    base_rejected = [
        row for row in rows
        if not row["corrupt_base_main_first_byte"]["recovery_parse"]["ok"]
        and row["corrupt_base_main_first_byte"]["recovery_parse"].get("error_type") == "BitstreamError"
    ]
    default_text_rejected = [
        row for row in text_rows
        if not row["corrupt_text_main_first_byte"]["default_parse"]["ok"]
    ]
    default_base_rejected = [
        row for row in rows
        if not row["corrupt_base_main_first_byte"]["default_parse"]["ok"]
    ]
    return {
        "stream_count": len(rows),
        "text_main_stream_count": len(text_rows),
        "default_rejects_text_main_corruption": len(default_text_rejected),
        "recovery_recovers_text_main_corruption": len(text_recovered),
        "default_rejects_base_main_corruption": len(default_base_rejected),
        "recovery_rejects_base_main_corruption": len(base_rejected),
        "all_text_main_corruptions_recovered": len(text_recovered) == len(text_rows),
        "all_base_main_corruptions_rejected": len(base_rejected) == len(rows),
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--stream-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--limit", type=int, default=None)
    args = parser.parse_args()

    paths = sorted(args.stream_dir.glob("*.oscr"))
    if args.limit is not None:
        paths = paths[: args.limit]
    if not paths:
        raise SystemExit(f"no .oscr streams found in {args.stream_dir}")

    rows = [audit_stream(path) for path in paths]
    output = {
        "stream_dir": str(args.stream_dir),
        "summary": summarize(rows),
        "rows": rows,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(output, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps(output["summary"], indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
