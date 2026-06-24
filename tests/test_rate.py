from pathlib import Path
import pytest

from oscarlic.rate import bpp_from_bytes, file_bpp, section_rate_breakdown


def test_bpp():
    assert bpp_from_bytes(100, 20, 10) == pytest.approx(4.0)


def test_file_bpp(tmp_path: Path):
    path = tmp_path / "stream.bin"
    path.write_bytes(b"x" * 25)
    result = file_bpp(path, 10, 10)
    assert result["actual_total_bytes"] == 25
    assert result["actual_total_bpp"] == pytest.approx(2.0)


def test_section_sum_guard():
    with pytest.raises(ValueError):
        section_rate_breakdown({"base": 10, "text": 2}, 10, 10, total_bytes=13)


def test_bad_dimensions():
    with pytest.raises(ValueError):
        bpp_from_bytes(1, 0, 10)
