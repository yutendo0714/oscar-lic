from pathlib import Path
import pytest

from oscarlic.bitstream import BitstreamError, Section, pack_container, parse_container


def make_stream():
    return pack_container(
        width=640,
        height=480,
        model_id=7,
        model_version=1,
        flags=1,
        sections=[
            Section(section_type=1, codec=0, flags=0, payload=b"hyper"),
            Section(section_type=2, codec=0, flags=0, payload=b"base-payload"),
            Section(section_type=10, codec=1, flags=0, payload=b"gate"),
            Section(section_type=12, codec=1, flags=0, payload=b"text-payload", dependency_mask=1),
        ],
    )


def test_round_trip():
    data = make_stream()
    parsed = parse_container(data)
    assert parsed.width == 640
    assert parsed.height == 480
    assert parsed.model_id == 7
    assert len(parsed.sections) == 4
    assert parsed.sections[1].payload == b"base-payload"
    assert parsed.total_bytes == len(data)
    assert parsed.bpp == pytest.approx(8 * len(data) / (640 * 480))


def test_corruption_detected():
    data = bytearray(make_stream())
    data[-8] ^= 0x01
    with pytest.raises(BitstreamError):
        parse_container(bytes(data))


def test_bad_magic():
    data = bytearray(make_stream())
    data[0:4] = b"FAIL"
    with pytest.raises(BitstreamError):
        parse_container(bytes(data), verify_crc=False)


def test_unprotected_length_guard():
    with pytest.raises(BitstreamError):
        pack_container(width=1, height=1, sections=[Section(1, 0, 0, b"x", unprotected_length=2)])
