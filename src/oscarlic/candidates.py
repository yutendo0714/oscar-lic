"""Candidate layout and gate-byte utilities for OSCAR allocation studies."""

from __future__ import annotations

from dataclasses import dataclass
from math import ceil
import struct
from typing import Iterable


_GATE_HEADER = struct.Struct("<HIII")
_COMPACT_CODEBOOK_PAIR = struct.Struct("<HB")


class GatePayloadError(ValueError):
    pass


class CompactCodebookPayloadError(ValueError):
    pass


@dataclass(frozen=True)
class LatentCandidate:
    index: int
    slice_index: int
    y0: int
    y1: int
    x0: int
    x1: int
    channel0: int
    channel1: int


@dataclass(frozen=True)
class LatentCandidateLayout:
    latent_height: int
    latent_width: int
    slice_num: int
    slice_channels: int
    tile_height: int
    tile_width: int
    channel_group_size: int | None = None

    def __post_init__(self) -> None:
        values = {
            "latent_height": self.latent_height,
            "latent_width": self.latent_width,
            "slice_num": self.slice_num,
            "slice_channels": self.slice_channels,
            "tile_height": self.tile_height,
            "tile_width": self.tile_width,
        }
        for name, value in values.items():
            if not isinstance(value, int) or value <= 0:
                raise ValueError(f"{name} must be a positive integer")
        if self.channel_group_size is not None:
            if not isinstance(self.channel_group_size, int) or self.channel_group_size <= 0:
                raise ValueError("channel_group_size must be a positive integer or None")
            if self.slice_channels % self.channel_group_size != 0:
                raise ValueError("slice_channels must be divisible by channel_group_size")

    @property
    def tiles_y(self) -> int:
        return ceil(self.latent_height / self.tile_height)

    @property
    def tiles_x(self) -> int:
        return ceil(self.latent_width / self.tile_width)

    @property
    def groups_per_slice(self) -> int:
        if self.channel_group_size is None:
            return 1
        return self.slice_channels // self.channel_group_size

    @property
    def candidate_count(self) -> int:
        return self.slice_num * self.groups_per_slice * self.tiles_y * self.tiles_x

    def iter_candidates(self) -> Iterable[LatentCandidate]:
        index = 0
        for slice_index in range(self.slice_num):
            for group in range(self.groups_per_slice):
                channel0 = group * (self.channel_group_size or self.slice_channels)
                channel1 = channel0 + (self.channel_group_size or self.slice_channels)
                for tile_y in range(self.tiles_y):
                    y0 = tile_y * self.tile_height
                    y1 = min(y0 + self.tile_height, self.latent_height)
                    for tile_x in range(self.tiles_x):
                        x0 = tile_x * self.tile_width
                        x1 = min(x0 + self.tile_width, self.latent_width)
                        yield LatentCandidate(index, slice_index, y0, y1, x0, x1, channel0, channel1)
                        index += 1


def gate_payload_bytes(
    *,
    num_candidates: int,
    selected_indices: Iterable[int],
    header_bytes: int = 14,
) -> dict:
    """Return simple gate payload byte counts for bitset vs index-list syntax.

    The 14-byte default header matches ``docs/BITSTREAM_SPEC.md``:
    uint16 candidate layout version, uint32 num candidates, uint32 selected
    count and uint32 probability model id.
    """
    if num_candidates <= 0:
        raise ValueError("num_candidates must be positive")
    selected = sorted(set(int(index) for index in selected_indices))
    if any(index < 0 or index >= num_candidates for index in selected):
        raise ValueError("selected index out of range")
    if header_bytes < 0:
        raise ValueError("header_bytes must be nonnegative")
    index_bytes = 2 if num_candidates <= 65535 else 4
    bitset_bytes = ceil(num_candidates / 8)
    list_bytes = len(selected) * index_bytes
    encoded_gate_bytes = min(bitset_bytes, list_bytes)
    mode = "bitset" if bitset_bytes <= list_bytes else "index_list"
    return {
        "num_candidates": num_candidates,
        "num_selected": len(selected),
        "index_bytes": index_bytes,
        "header_bytes": header_bytes,
        "bitset_bytes": bitset_bytes,
        "index_list_bytes": list_bytes,
        "encoded_gate_bytes": encoded_gate_bytes,
        "total_gate_payload_bytes": header_bytes + encoded_gate_bytes,
        "mode": mode,
    }


def encode_gate_payload(
    *,
    num_candidates: int,
    selected_indices: Iterable[int],
    layout_version: int = 1,
    probability_model_id: int = 0,
    mode: str = "auto",
) -> bytes:
    """Encode a concrete TEXT_GATE payload.

    The payload format follows the 14-byte header recorded in the bitstream
    spec. The body is either a dense little-endian bitset or a sparse sorted
    index list; ``auto`` chooses the shorter body with the bitset winning ties.
    """
    if num_candidates <= 0:
        raise GatePayloadError("num_candidates must be positive")
    if not 0 <= layout_version < (1 << 16):
        raise GatePayloadError("layout_version must fit uint16")
    if not 0 <= probability_model_id < (1 << 32):
        raise GatePayloadError("probability_model_id must fit uint32")
    selected = sorted(set(int(index) for index in selected_indices))
    if any(index < 0 or index >= num_candidates for index in selected):
        raise GatePayloadError("selected index out of range")
    if mode not in {"auto", "bitset", "index_list"}:
        raise GatePayloadError("mode must be auto, bitset or index_list")

    index_bytes = 2 if num_candidates <= 65535 else 4
    bitset_len = ceil(num_candidates / 8)
    list_len = len(selected) * index_bytes
    selected_mode = mode
    if selected_mode == "auto":
        selected_mode = "bitset" if bitset_len <= list_len else "index_list"

    if selected_mode == "bitset":
        body = bytearray(bitset_len)
        for index in selected:
            body[index // 8] |= 1 << (index % 8)
        encoded_body = bytes(body)
    else:
        item_struct = struct.Struct("<H" if index_bytes == 2 else "<I")
        encoded_body = b"".join(item_struct.pack(index) for index in selected)

    return _GATE_HEADER.pack(layout_version, num_candidates, len(selected), probability_model_id) + encoded_body


def decode_gate_payload(payload: bytes) -> dict:
    """Decode a TEXT_GATE payload produced by :func:`encode_gate_payload`."""
    if len(payload) < _GATE_HEADER.size:
        raise GatePayloadError("payload is shorter than gate header")
    layout_version, num_candidates, selected_count, probability_model_id = _GATE_HEADER.unpack_from(payload, 0)
    if num_candidates <= 0:
        raise GatePayloadError("num_candidates must be positive")
    body = payload[_GATE_HEADER.size:]
    index_bytes = 2 if num_candidates <= 65535 else 4
    bitset_len = ceil(num_candidates / 8)
    list_len = selected_count * index_bytes

    if len(body) == bitset_len:
        selected = []
        for index in range(num_candidates):
            if body[index // 8] & (1 << (index % 8)):
                selected.append(index)
        mode = "bitset"
        if len(selected) != selected_count:
            raise GatePayloadError("bitset selected count does not match header")
    elif len(body) == list_len:
        item_struct = struct.Struct("<H" if index_bytes == 2 else "<I")
        selected = [item_struct.unpack_from(body, offset)[0] for offset in range(0, len(body), index_bytes)]
        if len(set(selected)) != len(selected) or selected != sorted(selected):
            raise GatePayloadError("index list must be sorted and unique")
        if selected and (selected[0] < 0 or selected[-1] >= num_candidates):
            raise GatePayloadError("index list contains out-of-range entry")
        mode = "index_list"
    else:
        raise GatePayloadError("payload body length matches neither bitset nor index-list encoding")

    return {
        "layout_version": layout_version,
        "num_candidates": num_candidates,
        "num_selected": selected_count,
        "probability_model_id": probability_model_id,
        "mode": mode,
        "selected_indices": selected,
    }


def encode_compact_codebook_payload(candidate_indices: Iterable[int], code_indices: Iterable[int]) -> bytes:
    """Encode decoder-known codebook symbols as compact ``uint16,uint8`` pairs.

    This payload intentionally omits the v0 gate and codebook headers. The
    candidate layout and codebook id are treated as decoder-known model state
    for the research stub, matching the current codebook experiments.
    """
    candidates = [int(index) for index in candidate_indices]
    codes = [int(index) for index in code_indices]
    if len(candidates) != len(codes):
        raise CompactCodebookPayloadError("candidate and code index counts differ")
    if len(set(candidates)) != len(candidates) or candidates != sorted(candidates):
        raise CompactCodebookPayloadError("candidate indices must be sorted and unique")
    chunks = []
    for candidate_index, code_index in zip(candidates, codes):
        if candidate_index < 0 or candidate_index > 65535:
            raise CompactCodebookPayloadError("candidate indices must fit uint16")
        if code_index < 0 or code_index > 255:
            raise CompactCodebookPayloadError("code indices must fit uint8")
        chunks.append(_COMPACT_CODEBOOK_PAIR.pack(candidate_index, code_index))
    return b"".join(chunks)


def decode_compact_codebook_payload(payload: bytes, *, num_candidates: int | None = None) -> dict:
    """Decode compact decoder-known codebook ``uint16,uint8`` pairs."""
    if len(payload) % _COMPACT_CODEBOOK_PAIR.size != 0:
        raise CompactCodebookPayloadError("compact payload length is not a multiple of pair size")
    selected_indices = []
    code_indices = []
    for offset in range(0, len(payload), _COMPACT_CODEBOOK_PAIR.size):
        candidate_index, code_index = _COMPACT_CODEBOOK_PAIR.unpack_from(payload, offset)
        selected_indices.append(candidate_index)
        code_indices.append(code_index)
    if len(set(selected_indices)) != len(selected_indices) or selected_indices != sorted(selected_indices):
        raise CompactCodebookPayloadError("candidate indices must be sorted and unique")
    if num_candidates is not None and selected_indices and selected_indices[-1] >= num_candidates:
        raise CompactCodebookPayloadError("candidate index out of range")
    return {
        "mode": "compact_codebook_pairs_v0",
        "num_selected": len(selected_indices),
        "selected_indices": selected_indices,
        "code_indices": code_indices,
    }
