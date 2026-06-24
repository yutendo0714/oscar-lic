# AI Agent Operating Contract

## 1. Mission

本repositoryの目的は、画像の文字情報を低bitrateで保持しつつ、特定OCR modelへの過適合を避ける learned image compression 手法を研究・実装・評価することです。

主提案は **OSCAR-LIC: OCR-invariant Scalable Compression with Adaptive Rate Allocation** です。中心命題は、binaryなtext maskではなく、各latent unitに追加bitを与えたときの **marginal OCR utility per bit** を予測し、scalable enhancement streamを構成することです。

## 2. Authoritative read order

作業前に必ず次を読むこと。

1. `PROJECT_SPEC.md`: 研究問題、仮説、non-goal、成功条件
2. `STATUS.md`: 現在地と直近の結果
3. `DECISIONS.md`: 既に固定した判断
4. `NEGATIVE_RESULTS.md`: 再試行すべきでない失敗
5. `BLOCKERS.md`: 外部依存・未解決事項
6. `TASKS.md`: 優先順位付き作業
7. `evaluation/PROTOCOL.md`: 比較規則
8. `data/registry.yaml`: datasetの利用可否
9. `baselines/registry.yaml`: code、pin、license、再現状態
10. `experiments/EXPERIMENT_MATRIX.csv`: 実行すべき実験

矛盾時の優先順位は、`DECISIONS.md` > `PROJECT_SPEC.md` > machine-readable config > その他の説明文です。

## 3. Non-negotiable rules

### 3.1 Evidence

- author-reported result、独立再現値、推定値を混同しない。
- 数値を推測して埋めない。不明値は `unknown` または `null` とする。
- preprint、査読済み、標準文書、repository-onlyを区別する。
- 文献の主張を使う際は `literature/paper_registry.csv` のsource URLとstatusを確認する。
- 2026-06-24から30日以上経過している場合、最新文献・repository状態を更新してから「最新」と記述する。

### 3.2 Rate accounting

主結果のrateは必ず実ファイル長から計測する。

```text
R_total = R_base + R_hyper + R_gate + R_text + R_metadata + R_crc + R_fec
bpp = 8 * total_file_bytes / (H * W)
```

- entropy estimateのみの値を実bitstream bppとして扱わない。
- padding byte、stream header、shape、tile index、mask、文字列、位置、promptを無料にしない。
- semantic side-channelを使う方法はpure-image trackと同じ表で無条件に比較しない。

### 3.3 OCR protocol

- training teacherとheld-out evaluatorを分離する。
- oracle cropはupper-bound trackのみ。main trackはdetector込みend-to-end評価。
- ground-truth transcriptionがある場合、元画像に対するOCR出力を正解として扱わない。
- normalizationは `evaluation/OCR_NORMALIZATION.md` のprofile名で固定する。
- model、checkpoint、charset、input size、decoder、beam size、language packをmanifestに保存する。
- 同一OCRで改善して未知OCRで悪化した結果を「OCR preservation」と主張しない。

### 3.4 Data integrity

- raw datasetを変更しない。
- test splitをtraining、early stopping、hyperparameter selection、utility label生成に使わない。
- synthetic generatorのfont、background source、random seedを記録する。
- dataset間の重複とOCR pretrained data leakageを監査する。
- licenseが `manual_review_required` のdatasetは、承認記録なしに自動取得・再配布しない。

### 3.5 Reproducibility

全実験は以下に紐付ける。

- experiment ID
- hypothesis ID
- code commit
- config pathとconfig SHA256
- environment lock hash
- dataset manifest hash
- external repository pins
- checkpoint hash
- random seed
- hardware
- command
- stdout/stderr log
- metrics JSON
- actual bitstream archiveまたはそのmanifest

結果のみを手作業で表へ転記しない。元JSONからreportを生成する。

### 3.6 Research discipline

- baseline再現前に提案法のfull trainingを開始しない。
- oracle upper boundが成立しない機構を、predictorの容量増加だけで救済しようとしない。
- 新moduleを追加する前に、どの仮説を検証するのか明記する。
- 複数変更を同時に入れず、因果を特定できるablationを優先する。
- failed runを削除しない。`NEGATIVE_RESULTS.md` とexperiment registryへ記録する。
- visually appealingな例だけを選ばない。failure caseとconfusion matrixを保存する。

## 4. Research tracks

### Track A: Pure visual bitstream

文字列、OCR token、box座標のlossless side-channel送信は禁止。主要な学術比較はこのtrackです。

### Track B: Semantic side-channel allowed

文字列・layout metadata送信を許す。ただし全rate、encoder OCR cost、privacy leakageを計上します。PICD型手法との比較用です。

### Track C: Machine-only compressed-domain representation

RGB復元を必須とせず、OCRまたはdocument modelがlatentを直接読むtrackです。human reconstructionとのmulti-objective比較を別表で行います。

## 5. Mandatory stage gates

### G0 — Repository integrity

`python scripts/validate_repo.py` とunit testが合格。

### G1 — Base codec reproduction

実bitstreamのRD点が `baselines/expected_results.yaml` の許容条件内。author-reported BD-rateを直接再現できない場合でも、公開checkpointに対する内部回帰値を固定する。

### G2 — OCR evaluation validity

元画像、既知corruption、手計算例でCER/WER/NEDが一致し、detector-oracleとend-to-endの差が明示される。

### G3 — Single-teacher baseline

`R + lambda_D D + lambda_OCR L_OCR` を再現し、同一OCRとheld-out OCRの差を計測。

### G4 — Oracle utility feasibility

同じtotal bpp、同じtext ROI候補集合において、oracle marginal utility allocationがuniform text allocationとbinary mask allocationを上回る。上回らない場合は、utility predictor実装を停止し仮説を再定義する。

### G5 — Learned utility

random-utility control、mask confidence、uncertainty-only baselineをheld-out OCRで上回る。

### G6 — Scalable stream

base onlyが通常画像として復号可能で、enhancement追加時にrate単調増加とOCR performanceの概ね単調改善を示す。

### G7 — Cross-domain result

scene、screen、documentの3 domain中2つ以上で、強いpure-image baselineに対しBD-Rate@CERまたはBD-Rate@WERを改善。

### G8 — Submission readiness

全side bit計上、3 seed、confidence interval、complexity、failure analysis、license review、artifact smoke testが完了。

## 6. First-run procedure

1. `scripts/validate_repo.py` を実行。
2. `scripts/verify_environment.py` を実行しhardware/softwareを記録。
3. `literature/search_log.md` のfreshnessを確認。
4. `baselines/registry.yaml` のpriority P0 repositoryを取得・pin。
5. datasetを自動取得せず、licenseとmanual stepを確認。
6. toy imageでbppとOCR metricのsmoke test。
7. base codec 1 checkpoint × 5 imagesでencode/decode round trip。
8. `STATUS.md`、`TASKS.md`、experiment registryを更新。

## 7. How to choose the next task

優先順位は次です。

1. 再現性・評価の欠陥
2. stage gateを塞ぐblocker
3. center hypothesisを最小コストで反証できる実験
4. strong baseline
5. proposed method
6. optional generative refinement
7. paper polishing

GPUを使う前に、CPUで検証できる前提・metric・manifestを確認すること。

## 8. Required reporting after every run

experiment manifestに以下を追加する。

```yaml
experiment_id: E###
hypothesis_id: H#
status: completed | failed | invalid | interrupted
command: ...
code_commit: ...
config_sha256: ...
environment_sha256: ...
dataset_manifest_sha256: ...
seed: ...
hardware: ...
started_at: ...
finished_at: ...
metrics_path: ...
log_path: ...
checkpoint_path: ...
conclusion: ...
next_action: ...
```

invalid runはfailureと区別する。例: test leakage、wrong checkpoint、estimated bppのみ、metric bug。

## 9. Allowed autonomy

エージェントは、既存config内の小規模smoke test、unit test、registry validation、文献status更新を自律実行できます。以下は明示的な記録なしに行わないこと。

- test split変更
- primary metric変更
- baselineの公平性に影響するpreprocessing変更
- plaintext side-channel導入
- dataset license条件に触れるdownload・再配布
- Go/No-Go threshold変更
- negative result削除

必要な変更は `DECISIONS.md` にADR形式で提案・記録する。

## 10. Security and privacy

- API key、dataset credential、個人情報をcommitしない。
- OCR output、文字列stream、debug cropにPIIが含まれ得る。
- privacy benchmarkでは、utility/text latentから文字列を抽出するattackerを評価する。
- malicious checkpointを安全な形式として仮定しない。可能ならsafetensorsを使用し、hashを照合する。
- external scriptを実行する前に内容とlicenseを確認する。

## 11. Definition of done

研究directoryが完成したとは、単にcodeが動くことではなく、第三者が別環境で主要表を再生成でき、全rate・data split・model version・失敗条件・statistical uncertaintyを追跡できる状態を指します。
