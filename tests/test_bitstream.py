from pathlib import Path
import pytest

from oscarlic.bitstream import (
    BitstreamError,
    Section,
    pack_container,
    parse_container,
    parse_container_recovery,
)


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


def test_unknown_optional_section_can_be_skipped_for_base_only_decode():
    data = pack_container(
        width=32,
        height=16,
        model_id=1001,
        model_version=1,
        sections=[
            Section(section_type=2, codec=1001, flags=0, payload=b"base"),
            Section(section_type=65000, codec=7, flags=0, payload=b"future-optional"),
        ],
    )
    parsed = parse_container(data)
    base_sections = [section for section in parsed.sections if section.section_type == 2]
    assert len(base_sections) == 1
    assert base_sections[0].payload == b"base"
    assert parsed.total_bytes == len(data)


def test_recovery_rejects_corrupt_optional_section_but_keeps_base():
    data = bytearray(make_stream())
    parsed = parse_container(bytes(data))
    text_section = [section for section in parsed.sections if section.section_type == 12][0]
    data[text_section.offset] ^= 0x01

    with pytest.raises(BitstreamError):
        parse_container(bytes(data))

    recovered = parse_container_recovery(bytes(data))
    recovered_base = [section for section in recovered.sections if section.section_type == 2]
    assert recovered.file_crc_ok is False
    assert len(recovered_base) == 1
    assert recovered_base[0].payload == b"base-payload"
    assert [section.section_type for section in recovered.sections] == [1, 2, 10]
    assert len(recovered.rejected_sections) == 1
    assert recovered.rejected_sections[0].index == 3
    assert recovered.rejected_sections[0].section_type == 12
    assert recovered.rejected_sections[0].error == "payload CRC mismatch"


def test_recovery_rejects_corrupt_required_base_section():
    data = bytearray(make_stream())
    parsed = parse_container(bytes(data))
    base_section = [section for section in parsed.sections if section.section_type == 2][0]
    data[base_section.offset] ^= 0x01

    with pytest.raises(BitstreamError, match="required section"):
        parse_container_recovery(bytes(data))
