from pathlib import Path
import json
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]


def test_text_metric_cli(tmp_path: Path):
    input_path = tmp_path / "pairs.jsonl"
    input_path.write_text(json.dumps({"reference": "ABC", "hypothesis": "ADC"}) + "\n", encoding="utf-8")
    completed = subprocess.run(
        [sys.executable, str(ROOT / "scripts/compute_text_metrics.py"), str(input_path)],
        cwd=ROOT,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=False,
    )
    assert completed.returncode == 0, completed.stderr
    result = json.loads(completed.stdout)
    assert result["cer_micro"] == 1 / 3
