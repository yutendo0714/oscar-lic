import json
from pathlib import Path

from scripts.build_interaction_smoke_policies import main as build_interaction_main
from scripts.filter_assignment_policy_rows import main as filter_policy_main


def write_jsonl(path: Path, rows: list[dict]) -> None:
    path.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")


def test_filter_assignment_policy_rows(tmp_path, monkeypatch):
    source = tmp_path / "policy.jsonl"
    output = tmp_path / "filtered.jsonl"
    write_jsonl(
        source,
        [
            {
                "source_index": 0,
                "candidate_index": 1,
                "code_index": 4,
                "nearest_code": 3,
                "seed": 1,
                "real_seed": 2,
                "policy": "keep",
                "parseq_delta_vs_nearest": 0,
                "tesseract_delta_vs_nearest": -1,
                "gate_threshold": 0.5,
                "gate_model_seed": 7,
                "score_model_seed": 0,
            },
            {
                "source_index": 1,
                "candidate_index": 2,
                "code_index": 3,
                "nearest_code": 3,
                "seed": 0,
                "real_seed": 2,
                "policy": "keep",
                "parseq_delta_vs_nearest": 0,
                "tesseract_delta_vs_nearest": 0,
            },
        ],
    )
    monkeypatch.setattr(
        "sys.argv",
        [
            "filter_assignment_policy_rows.py",
            "--input",
            str(source),
            "--output",
            str(output),
            "--seed",
            "1",
            "--real-seed",
            "2",
            "--policy",
            "keep",
        ],
    )
    assert filter_policy_main() == 0
    rows = [json.loads(line) for line in output.read_text(encoding="utf-8").splitlines()]
    assert len(rows) == 1
    assert rows[0]["source_index"] == 0
    summary = json.loads((tmp_path / "filtered.jsonl.summary.json").read_text(encoding="utf-8"))
    assert summary["groups"] == 1
    assert summary["changed_groups"] == 1
    assert summary["tesseract_delta_sum_vs_nearest"] == -1


def test_build_interaction_smoke_policies_selects_distinct_candidates(tmp_path, monkeypatch):
    split = tmp_path / "split.jsonl"
    table = tmp_path / "candidates.jsonl"
    output_split = tmp_path / "smoke.jsonl"
    prefix = tmp_path / "policy"
    write_jsonl(
        split,
        [
            {"image_path": "/tmp/a.png", "text": "A", "source": "toy", "split": "train"},
            {"image_path": "/tmp/b.png", "text": "B", "source": "toy", "split": "train"},
        ],
    )
    write_jsonl(
        table,
        [
            {
                "source_index": 0,
                "candidate_index": 1,
                "multi_teacher_delta_distance": -1,
                "tesseract_delta_distance": -1,
                "parseq_delta_distance": 0,
                "label_no_evaluator_worsens": 1,
                "source_image": "/tmp/a.png",
                "reference": "A",
            },
            {
                "source_index": 0,
                "candidate_index": 2,
                "multi_teacher_delta_distance": -1,
                "tesseract_delta_distance": 0,
                "parseq_delta_distance": -1,
                "label_no_evaluator_worsens": 1,
                "source_image": "/tmp/a.png",
                "reference": "A",
            },
            {
                "source_index": 1,
                "candidate_index": 3,
                "multi_teacher_delta_distance": -1,
                "tesseract_delta_distance": -1,
                "parseq_delta_distance": 0,
                "label_no_evaluator_worsens": 1,
                "source_image": "/tmp/b.png",
                "reference": "B",
            },
        ],
    )
    monkeypatch.setattr(
        "sys.argv",
        [
            "build_interaction_smoke_policies.py",
            "--candidate-table",
            str(table),
            "--source-split",
            str(split),
            "--output-split",
            str(output_split),
            "--output-policy-prefix",
            str(prefix),
            "--max-sources",
            "1",
            "--require-no-evaluator-worsens",
        ],
    )
    assert build_interaction_main() == 0
    pair_rows = [json.loads(line) for line in (tmp_path / "policy_pair_ab.jsonl").read_text().splitlines()]
    assert [row["candidate_index"] for row in pair_rows] == [1, 2]
    assert all(row["source_index"] == 0 for row in pair_rows)
    assert len(output_split.read_text(encoding="utf-8").splitlines()) == 1
