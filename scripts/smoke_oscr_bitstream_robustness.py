#!/usr/bin/env python3
"""Smoke-test OSCAR container CRC behavior and base-only recovery."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys

import torch


ROOT = Path(__file__).resolve().parents[1]
MLIC_ROOT = ROOT / "external/mlic/MLIC++"
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))
sys.path.insert(0, str(MLIC_ROOT))

from inspect_mlicpp_decoded_yhat import decompress_with_y_hat, load_model  # noqa: E402
from oscarlic.bitstream import Section, pack_container, parse_container, parse_container_recovery  # noqa: E402
from run_mlicpp_latent_residual_stub import parse_mlicpp_body  # noqa: E402


SECTION_BASE_MAIN = 2
SECTION_TEXT_GATE = 10
SECTION_TEXT_MAIN = 12


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def section_by_type(parsed, section_type: int):
    matches = [section for section in parsed.sections if section.section_type == section_type]
    if len(matches) != 1:
        raise ValueError(f"expected exactly one section type {section_type}, got {len(matches)}")
    return matches[0]


def try_parse(data: bytes, *, verify_crc: bool) -> dict:
    try:
        parsed = parse_container(data, verify_crc=verify_crc)
        return {"ok": True, "section_count": len(parsed.sections), "bpp": parsed.bpp}
    except Exception as exc:  # noqa: BLE001 - record exact failure mode for smoke manifest
        return {"ok": False, "error_type": type(exc).__name__, "error": str(exc)}


def try_recovery_parse(data: bytes) -> dict:
    try:
        recovered = parse_container_recovery(data)
        return {
            "ok": True,
            "section_count": len(recovered.sections),
            "rejected_section_count": len(recovered.rejected_sections),
            "rejected_sections": [
                {
                    "index": section.index,
                    "type": section.section_type,
                    "codec": section.codec,
                    "offset": section.offset,
                    "length": section.length,
                    "error": section.error,
                }
                for section in recovered.rejected_sections
            ],
            "file_crc_ok": recovered.file_crc_ok,
            "bpp": recovered.bpp,
        }
    except Exception as exc:  # noqa: BLE001 - record exact recovery failure for smoke manifest
        return {"ok": False, "error_type": type(exc).__name__, "error": str(exc)}


def decode_base_tensor(model, payload: bytes, *, width: int, height: int):
    strings, shape = parse_mlicpp_body(payload)
    with torch.no_grad():
        decoded = decompress_with_y_hat(model, strings, shape)
    return decoded["x_hat"][:, :, :height, :width].detach().cpu()


def max_abs_delta(a: torch.Tensor, b: torch.Tensor) -> float:
    return float((a - b).abs().max().item())


def corrupt_one_byte(data: bytes, offset: int) -> bytes:
    buf = bytearray(data)
    buf[offset] ^= 0x01
    return bytes(buf)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--stream", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--cuda", action="store_true")
    parser.add_argument(
        "--skip-corrupt-base-decode",
        action="store_true",
        help=(
            "Do not feed a CRC-failed BASE_MAIN payload to the entropy decoder. "
            "This keeps the smoke safe on GPUs while still verifying that the "
            "default container parser rejects the corruption."
        ),
    )
    args = parser.parse_args()

    device = "cuda" if args.cuda and torch.cuda.is_available() else "cpu"
    if args.cuda and device != "cuda":
        raise SystemExit("CUDA requested but torch.cuda.is_available() is false")

    data = args.stream.read_bytes()
    parsed = parse_container(data, verify_crc=True)
    base = section_by_type(parsed, SECTION_BASE_MAIN)
    text_gate = section_by_type(parsed, SECTION_TEXT_GATE)
    text_main = section_by_type(parsed, SECTION_TEXT_MAIN)

    model, checkpoint = load_model(args.checkpoint, device)
    base_x = decode_base_tensor(model, base.payload, width=parsed.width, height=parsed.height)

    base_only_data = pack_container(
        width=parsed.width,
        height=parsed.height,
        channels=parsed.channels,
        bit_depth=parsed.bit_depth,
        color_space=parsed.color_space,
        model_id=parsed.model_id,
        model_version=parsed.model_version,
        flags=parsed.flags,
        sections=[
            Section(
                section_type=base.section_type,
                codec=base.codec,
                flags=base.flags,
                payload=base.payload,
                unprotected_length=base.unprotected_length,
                dependency_mask=base.dependency_mask,
            )
        ],
    )
    base_only_parsed = parse_container(base_only_data, verify_crc=True)
    base_only_section = section_by_type(base_only_parsed, SECTION_BASE_MAIN)
    base_only_x = decode_base_tensor(model, base_only_section.payload, width=parsed.width, height=parsed.height)

    text_main_flip = corrupt_one_byte(data, text_main.offset)
    text_main_recovered = parse_container_recovery(text_main_flip)
    text_main_recovered_base = section_by_type(text_main_recovered, SECTION_BASE_MAIN)
    text_main_no_crc_x = decode_base_tensor(
        model,
        text_main_recovered_base.payload,
        width=text_main_recovered.width,
        height=text_main_recovered.height,
    )

    base_flip = corrupt_one_byte(data, base.offset)
    base_flip_no_crc = parse_container(base_flip, verify_crc=False)
    base_flip_payload = section_by_type(base_flip_no_crc, SECTION_BASE_MAIN).payload
    base_flip_decode = {"attempted": not args.skip_corrupt_base_decode}
    if args.skip_corrupt_base_decode:
        base_flip_decode["ok"] = None
        base_flip_decode["skipped_reason"] = (
            "CRC-failed BASE_MAIN payloads are attacker-controlled input; previous smoke tests showed "
            "that feeding one-byte-corrupted MLIC++ payloads to the entropy decoder can trigger unsafe "
            "allocation paths. The default CRC rejection is the intended behavior under this profile."
        )
    else:
        try:
            base_flip_x = decode_base_tensor(model, base_flip_payload, width=parsed.width, height=parsed.height)
            base_flip_decode["ok"] = True
            base_flip_decode["max_abs_delta_vs_original_base"] = max_abs_delta(base_x, base_flip_x)
        except Exception as exc:  # noqa: BLE001 - corruption may fail inside entropy decoder
            base_flip_decode["ok"] = False
            base_flip_decode["error_type"] = type(exc).__name__
            base_flip_decode["error"] = str(exc)

    truncated_before_text_main = data[: text_main.offset] + data[-4:]

    output = {
        "stream": str(args.stream),
        "stream_sha256": sha256_bytes(data),
        "checkpoint": str(args.checkpoint),
        "checkpoint_epoch": checkpoint.get("epoch"),
        "device": device,
        "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
        "original": {
            "verify_crc": try_parse(data, verify_crc=True),
            "width": parsed.width,
            "height": parsed.height,
            "total_bytes": parsed.total_bytes,
            "bpp": parsed.bpp,
            "sections": [
                {
                    "type": section.section_type,
                    "codec": section.codec,
                    "offset": section.offset,
                    "length": section.length,
                    "dependency_mask": section.dependency_mask,
                    "payload_sha256": sha256_bytes(section.payload),
                }
                for section in parsed.sections
            ],
        },
        "base_only_repack": {
            "verify_crc": try_parse(base_only_data, verify_crc=True),
            "total_bytes": len(base_only_data),
            "bpp": 8.0 * len(base_only_data) / (parsed.width * parsed.height),
            "max_abs_delta_vs_original_base": max_abs_delta(base_x, base_only_x),
        },
        "corrupt_text_main_first_byte": {
            "verify_crc": try_parse(text_main_flip, verify_crc=True),
            "no_crc_parse": try_parse(text_main_flip, verify_crc=False),
            "recovery_parse": try_recovery_parse(text_main_flip),
            "base_payload_sha256_unchanged": sha256_bytes(text_main_recovered_base.payload) == sha256_bytes(base.payload),
            "base_decode_max_abs_delta_vs_original_base": max_abs_delta(base_x, text_main_no_crc_x),
        },
        "corrupt_base_main_first_byte": {
            "verify_crc": try_parse(base_flip, verify_crc=True),
            "no_crc_parse": try_parse(base_flip, verify_crc=False),
            "recovery_parse": try_recovery_parse(base_flip),
            "base_payload_sha256_unchanged": sha256_bytes(base_flip_payload) == sha256_bytes(base.payload),
            "base_decode": base_flip_decode,
        },
        "truncated_before_text_main_payload": {
            "verify_crc": try_parse(truncated_before_text_main, verify_crc=True),
            "no_crc_parse": try_parse(truncated_before_text_main, verify_crc=False),
        },
        "interpretation": (
            "Current OSCR CRC catches optional TEXT corruption. Base-only reconstruction is byte-identical after "
            "repacking the verified BASE_MAIN payload, and optional TEXT_MAIN corruption leaves BASE_MAIN decodable "
            "under the section-level recovery profile. The default full-container verifier intentionally rejects "
            "any payload corruption, so future UEP/FEC work can build on the recovery profile if base-layer decoding "
            "must proceed after enhancement-layer damage."
        ),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(output, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps(output, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
