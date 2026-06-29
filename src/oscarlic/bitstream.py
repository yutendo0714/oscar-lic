"""Minimal OSCAR research container implementation for accounting tests.

This module implements the version 0.1 container from ``docs/BITSTREAM_SPEC.md``.
It is not an entropy codec and should not be confused with the future neural
codec implementation.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import struct
import zlib

MAGIC = b"OSCR"
MAJOR = 0
MINOR = 1
HEADER_STRUCT = struct.Struct("<4sBBHII4BIIHHQI")
SECTION_STRUCT = struct.Struct("<HHIQQQII")
FILE_CRC_STRUCT = struct.Struct("<I")
assert HEADER_STRUCT.size == 44
assert SECTION_STRUCT.size == 40


class BitstreamError(ValueError):
    pass


@dataclass(frozen=True)
class Section:
    section_type: int
    codec: int
    flags: int
    payload: bytes
    unprotected_length: int | None = None
    dependency_mask: int = 0


@dataclass(frozen=True)
class ParsedSection:
    section_type: int
    codec: int
    flags: int
    offset: int
    length: int
    unprotected_length: int
    payload_crc32: int
    dependency_mask: int
    payload: bytes


@dataclass(frozen=True)
class ParsedContainer:
    width: int
    height: int
    channels: int
    bit_depth: int
    color_space: int
    model_id: int
    model_version: int
    flags: int
    sections: tuple[ParsedSection, ...]
    total_bytes: int

    @property
    def bpp(self) -> float:
        return 8.0 * self.total_bytes / (self.width * self.height)


@dataclass(frozen=True)
class SectionRecoveryIssue:
    index: int
    section_type: int
    codec: int
    offset: int
    length: int
    error: str


@dataclass(frozen=True)
class RecoveredContainer:
    width: int
    height: int
    channels: int
    bit_depth: int
    color_space: int
    model_id: int
    model_version: int
    flags: int
    sections: tuple[ParsedSection, ...]
    rejected_sections: tuple[SectionRecoveryIssue, ...]
    total_bytes: int
    file_crc_ok: bool

    @property
    def bpp(self) -> float:
        return 8.0 * self.total_bytes / (self.width * self.height)


def _validate_u(value: int, bits: int, name: str) -> None:
    if not isinstance(value, int) or value < 0 or value >= (1 << bits):
        raise BitstreamError(f"{name} must be uint{bits}, got {value!r}")


def pack_container(
    *,
    width: int,
    height: int,
    sections: list[Section],
    channels: int = 3,
    bit_depth: int = 8,
    color_space: int = 0,
    model_id: int = 0,
    model_version: int = 0,
    flags: int = 0,
) -> bytes:
    if width <= 0 or height <= 0:
        raise BitstreamError("width and height must be positive")
    _validate_u(width, 32, "width")
    _validate_u(height, 32, "height")
    _validate_u(channels, 8, "channels")
    _validate_u(bit_depth, 8, "bit_depth")
    _validate_u(color_space, 8, "color_space")
    _validate_u(model_id, 32, "model_id")
    _validate_u(model_version, 32, "model_version")
    _validate_u(flags, 16, "flags")
    _validate_u(len(sections), 16, "section_count")

    header_bytes = HEADER_STRUCT.size + SECTION_STRUCT.size * len(sections)
    offset = header_bytes
    table_rows: list[bytes] = []
    payloads: list[bytes] = []
    for section in sections:
        payload = bytes(section.payload)
        unprotected = len(payload) if section.unprotected_length is None else section.unprotected_length
        if unprotected < 0 or unprotected > len(payload):
            raise BitstreamError("unprotected_length must be within payload length")
        crc = zlib.crc32(payload) & 0xFFFFFFFF
        table_rows.append(
            SECTION_STRUCT.pack(
                section.section_type,
                section.codec,
                section.flags,
                offset,
                len(payload),
                unprotected,
                crc,
                section.dependency_mask,
            )
        )
        payloads.append(payload)
        offset += len(payload)

    total_bytes = offset + FILE_CRC_STRUCT.size
    table = b"".join(table_rows)
    header_zero_crc = HEADER_STRUCT.pack(
        MAGIC, MAJOR, MINOR, flags, width, height, channels, bit_depth,
        color_space, 0, model_id, model_version, len(sections), header_bytes,
        total_bytes, 0,
    )
    header_crc = zlib.crc32(header_zero_crc + table) & 0xFFFFFFFF
    header = HEADER_STRUCT.pack(
        MAGIC, MAJOR, MINOR, flags, width, height, channels, bit_depth,
        color_space, 0, model_id, model_version, len(sections), header_bytes,
        total_bytes, header_crc,
    )
    without_file_crc = header + table + b"".join(payloads)
    file_crc = zlib.crc32(without_file_crc) & 0xFFFFFFFF
    return without_file_crc + FILE_CRC_STRUCT.pack(file_crc)


def parse_container(data: bytes, *, verify_crc: bool = True, max_pixels: int = 1_000_000_000) -> ParsedContainer:
    if len(data) < HEADER_STRUCT.size + FILE_CRC_STRUCT.size:
        raise BitstreamError("file is shorter than minimum container")
    fields = HEADER_STRUCT.unpack_from(data, 0)
    (
        magic, major, minor, flags, width, height, channels, bit_depth,
        color_space, reserved0, model_id, model_version, section_count,
        header_bytes, total_bytes, header_crc,
    ) = fields
    if magic != MAGIC:
        raise BitstreamError("bad magic")
    if major != MAJOR:
        raise BitstreamError(f"unsupported major version {major}")
    if minor > MINOR:
        raise BitstreamError(f"unsupported newer minor version {minor}")
    if reserved0 != 0:
        raise BitstreamError("reserved header byte is nonzero")
    if width == 0 or height == 0 or width * height > max_pixels:
        raise BitstreamError("invalid or excessive image dimensions")
    expected_header_bytes = HEADER_STRUCT.size + section_count * SECTION_STRUCT.size
    if header_bytes != expected_header_bytes:
        raise BitstreamError("header_bytes does not match section count")
    if total_bytes != len(data):
        raise BitstreamError("declared total_bytes does not match file length")
    if header_bytes + FILE_CRC_STRUCT.size > len(data):
        raise BitstreamError("section table exceeds file")

    if verify_crc:
        header_zero = HEADER_STRUCT.pack(
            magic, major, minor, flags, width, height, channels, bit_depth,
            color_space, reserved0, model_id, model_version, section_count,
            header_bytes, total_bytes, 0,
        )
        table = data[HEADER_STRUCT.size:header_bytes]
        if (zlib.crc32(header_zero + table) & 0xFFFFFFFF) != header_crc:
            raise BitstreamError("header CRC mismatch")
        expected_file_crc = FILE_CRC_STRUCT.unpack_from(data, len(data) - 4)[0]
        if (zlib.crc32(data[:-4]) & 0xFFFFFFFF) != expected_file_crc:
            raise BitstreamError("file CRC mismatch")

    sections: list[ParsedSection] = []
    occupied: list[tuple[int, int]] = []
    for index in range(section_count):
        start = HEADER_STRUCT.size + index * SECTION_STRUCT.size
        values = SECTION_STRUCT.unpack_from(data, start)
        stype, codec, sflags, offset, length, unprotected, crc, dependency = values
        end = offset + length
        if offset < header_bytes or end > len(data) - FILE_CRC_STRUCT.size or end < offset:
            raise BitstreamError(f"section {index} is out of range")
        if unprotected > length:
            raise BitstreamError(f"section {index} has invalid unprotected length")
        for other_start, other_end in occupied:
            if not (end <= other_start or offset >= other_end):
                raise BitstreamError(f"section {index} overlaps another section")
        occupied.append((offset, end))
        payload = data[offset:end]
        if verify_crc and (zlib.crc32(payload) & 0xFFFFFFFF) != crc:
            raise BitstreamError(f"section {index} payload CRC mismatch")
        sections.append(ParsedSection(stype, codec, sflags, offset, length, unprotected, crc, dependency, payload))

    return ParsedContainer(
        width=width, height=height, channels=channels, bit_depth=bit_depth,
        color_space=color_space, model_id=model_id, model_version=model_version,
        flags=flags, sections=tuple(sections), total_bytes=total_bytes,
    )


def parse_container_recovery(
    data: bytes,
    *,
    required_section_types: tuple[int, ...] = (2,),
    max_pixels: int = 1_000_000_000,
) -> RecoveredContainer:
    """Parse a container for a conservative section-level recovery profile.

    The default parser remains fail-closed: any file CRC mismatch rejects the
    stream. This helper is intentionally narrower. It verifies the header and
    section table first, then keeps only sections whose payload CRC matches. If
    a required payload, such as ``BASE_MAIN`` type 2, is corrupt, recovery
    fails instead of returning attacker-controlled bytes to a decoder.
    """

    if len(data) < HEADER_STRUCT.size + FILE_CRC_STRUCT.size:
        raise BitstreamError("file is shorter than minimum container")
    fields = HEADER_STRUCT.unpack_from(data, 0)
    (
        magic, major, minor, flags, width, height, channels, bit_depth,
        color_space, reserved0, model_id, model_version, section_count,
        header_bytes, total_bytes, header_crc,
    ) = fields
    if magic != MAGIC:
        raise BitstreamError("bad magic")
    if major != MAJOR:
        raise BitstreamError(f"unsupported major version {major}")
    if minor > MINOR:
        raise BitstreamError(f"unsupported newer minor version {minor}")
    if reserved0 != 0:
        raise BitstreamError("reserved header byte is nonzero")
    if width == 0 or height == 0 or width * height > max_pixels:
        raise BitstreamError("invalid or excessive image dimensions")
    expected_header_bytes = HEADER_STRUCT.size + section_count * SECTION_STRUCT.size
    if header_bytes != expected_header_bytes:
        raise BitstreamError("header_bytes does not match section count")
    if total_bytes != len(data):
        raise BitstreamError("declared total_bytes does not match file length")
    if header_bytes + FILE_CRC_STRUCT.size > len(data):
        raise BitstreamError("section table exceeds file")

    header_zero = HEADER_STRUCT.pack(
        magic, major, minor, flags, width, height, channels, bit_depth,
        color_space, reserved0, model_id, model_version, section_count,
        header_bytes, total_bytes, 0,
    )
    table = data[HEADER_STRUCT.size:header_bytes]
    if (zlib.crc32(header_zero + table) & 0xFFFFFFFF) != header_crc:
        raise BitstreamError("header CRC mismatch")
    expected_file_crc = FILE_CRC_STRUCT.unpack_from(data, len(data) - 4)[0]
    file_crc_ok = (zlib.crc32(data[:-4]) & 0xFFFFFFFF) == expected_file_crc

    required = set(required_section_types)
    sections: list[ParsedSection] = []
    rejected: list[SectionRecoveryIssue] = []
    occupied: list[tuple[int, int]] = []
    for index in range(section_count):
        start = HEADER_STRUCT.size + index * SECTION_STRUCT.size
        values = SECTION_STRUCT.unpack_from(data, start)
        stype, codec, sflags, offset, length, unprotected, crc, dependency = values
        end = offset + length
        if offset < header_bytes or end > len(data) - FILE_CRC_STRUCT.size or end < offset:
            raise BitstreamError(f"section {index} is out of range")
        if unprotected > length:
            raise BitstreamError(f"section {index} has invalid unprotected length")
        for other_start, other_end in occupied:
            if not (end <= other_start or offset >= other_end):
                raise BitstreamError(f"section {index} overlaps another section")
        occupied.append((offset, end))
        payload = data[offset:end]
        if (zlib.crc32(payload) & 0xFFFFFFFF) != crc:
            error = "payload CRC mismatch"
            if stype in required:
                raise BitstreamError(f"required section {index} payload CRC mismatch")
            rejected.append(SectionRecoveryIssue(index, stype, codec, offset, length, error))
            continue
        sections.append(ParsedSection(stype, codec, sflags, offset, length, unprotected, crc, dependency, payload))

    return RecoveredContainer(
        width=width, height=height, channels=channels, bit_depth=bit_depth,
        color_space=color_space, model_id=model_id, model_version=model_version,
        flags=flags, sections=tuple(sections), rejected_sections=tuple(rejected),
        total_bytes=total_bytes, file_crc_ok=file_crc_ok,
    )


def write_container(path: str | Path, **kwargs) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(pack_container(**kwargs))
    return path


def read_container(path: str | Path, *, verify_crc: bool = True) -> ParsedContainer:
    return parse_container(Path(path).read_bytes(), verify_crc=verify_crc)
