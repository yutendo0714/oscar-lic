import pytest

from oscarlic.candidates import (
    CompactCodebookPayloadError,
    LatentCandidateLayout,
    decode_compact_codebook_payload,
    decode_gate_payload,
    encode_compact_codebook_payload,
    encode_gate_payload,
    gate_payload_bytes,
)


def test_latent_candidate_layout_counts_partial_tiles():
    layout = LatentCandidateLayout(
        latent_height=5,
        latent_width=9,
        slice_num=10,
        slice_channels=32,
        tile_height=4,
        tile_width=4,
    )
    candidates = list(layout.iter_candidates())
    assert layout.tiles_y == 2
    assert layout.tiles_x == 3
    assert layout.candidate_count == 60
    assert len(candidates) == 60
    assert candidates[0].slice_index == 0
    assert candidates[-1].slice_index == 9
    assert candidates[-1].y0 == 4
    assert candidates[-1].y1 == 5
    assert candidates[-1].x0 == 8
    assert candidates[-1].x1 == 9


def test_latent_candidate_layout_channel_groups():
    layout = LatentCandidateLayout(
        latent_height=8,
        latent_width=8,
        slice_num=2,
        slice_channels=32,
        channel_group_size=8,
        tile_height=4,
        tile_width=4,
    )
    candidates = list(layout.iter_candidates())
    assert layout.groups_per_slice == 4
    assert layout.candidate_count == 32
    assert candidates[0].channel0 == 0
    assert candidates[0].channel1 == 8
    assert candidates[4].channel0 == 8
    assert candidates[4].channel1 == 16


def test_gate_payload_chooses_sparse_index_list():
    result = gate_payload_bytes(num_candidates=1000, selected_indices=[1, 5, 9])
    assert result["mode"] == "index_list"
    assert result["index_bytes"] == 2
    assert result["encoded_gate_bytes"] == 6
    assert result["total_gate_payload_bytes"] == 20


def test_gate_payload_chooses_bitset_for_dense_selection():
    result = gate_payload_bytes(num_candidates=100, selected_indices=range(90))
    assert result["mode"] == "bitset"
    assert result["bitset_bytes"] == 13
    assert result["total_gate_payload_bytes"] == 27


def test_gate_payload_rejects_out_of_range():
    with pytest.raises(ValueError):
        gate_payload_bytes(num_candidates=4, selected_indices=[4])


def test_gate_payload_round_trip_sparse_index_list():
    payload = encode_gate_payload(
        num_candidates=1000,
        selected_indices=[3, 7, 999],
        layout_version=2,
        probability_model_id=11,
    )
    decoded = decode_gate_payload(payload)
    assert decoded["layout_version"] == 2
    assert decoded["probability_model_id"] == 11
    assert decoded["mode"] == "index_list"
    assert decoded["selected_indices"] == [3, 7, 999]
    assert len(payload) == gate_payload_bytes(
        num_candidates=1000,
        selected_indices=[3, 7, 999],
    )["total_gate_payload_bytes"]


def test_gate_payload_round_trip_dense_bitset():
    selected = list(range(15))
    payload = encode_gate_payload(num_candidates=16, selected_indices=selected)
    decoded = decode_gate_payload(payload)
    assert decoded["mode"] == "bitset"
    assert decoded["selected_indices"] == selected


def test_compact_codebook_payload_round_trip():
    payload = encode_compact_codebook_payload([3, 7, 999], [1, 0, 63])
    assert len(payload) == 9
    decoded = decode_compact_codebook_payload(payload, num_candidates=1000)
    assert decoded["mode"] == "compact_codebook_pairs_v0"
    assert decoded["selected_indices"] == [3, 7, 999]
    assert decoded["code_indices"] == [1, 0, 63]


def test_compact_codebook_payload_rejects_unsorted_or_out_of_range():
    with pytest.raises(CompactCodebookPayloadError):
        encode_compact_codebook_payload([7, 3], [1, 2])
    with pytest.raises(CompactCodebookPayloadError):
        encode_compact_codebook_payload([3], [256])
    payload = encode_compact_codebook_payload([3], [1])
    with pytest.raises(CompactCodebookPayloadError):
        decode_compact_codebook_payload(payload, num_candidates=3)
