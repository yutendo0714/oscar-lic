# OSCAR-LIC Research Repository

**Snapshot date:** 2026-06-24  
**Repository version:** 0.1.0-bootstrap  
**Research target:** OCR-model-invariant, scalable learned image compression with utility-guided bit allocation

このdirectoryは、AIエージェントまたは研究者が、研究目的を質問し直さずに、環境確認、baseline再現、OCR評価、提案法実装、ablation、統計解析、論文執筆まで開始できるよう設計した **self-bootstrapping research repository** です。

## 最初に読む順序

1. `AGENTS.md`
2. `PROJECT_SPEC.md`
3. `STATUS.md`
4. `DECISIONS.md`
5. `TASKS.md`
6. `docs/architecture.md`
7. `evaluation/PROTOCOL.md`
8. `baselines/registry.yaml`
9. `data/registry.yaml`
10. `experiments/EXPERIMENT_MATRIX.csv`

## このsnapshotに含まれるもの

- LIC、generative compression、semantic text-guided compression、ICM、OCR-aware compression、RAW、robustnessの体系的survey（58 structured records）
- 公式・準公式実装のregistry、2026-06-24時点の一部branch HEAD pin
- datasetの用途、license risk、分割、normalization、漏洩防止規則
- OSCAR-LICのarchitecture、loss、utility oracle、bitstream案
- baseline再現手順、実験行列、ablation、Go/No-Go gate
- CER/WER、total bpp、manifest生成などの実行可能な小規模utility
- agent作業規約、状態管理、negative result、decision logのtemplate

## 意図的に同梱していないもの

論文PDF、dataset本体、外部repository本体、checkpoint、font file、認証情報は、著作権・license・容量・更新性の理由から同梱していません。代わりに、取得元、license確認状態、expected path、checksum登録手順をregistryへ記載しています。

## 最初の実行

```bash
cd oscar_lic_research_repo
python scripts/validate_repo.py
python scripts/render_literature.py --check
python scripts/validate_dataset_manifest.py data/toy/manifest.jsonl --root data/toy --require-hash
python -m pytest -q
python scripts/verify_environment.py
```

外部repositoryを取得できる環境では、次を実行します。

```bash
python scripts/bootstrap_external_repos.py --dry-run
python scripts/bootstrap_external_repos.py --selected compressai mlic parseq
```

`--dry-run`なしで取得したcommitは `baselines/locks/resolved_repositories.json` に保存されます。registryに既にpinがある場合は、そのcommitをcheckoutします。

## 研究の最小開始条件

提案法の大規模学習へ進む前に、次を満たしてください。

- MLIC++または選択したbase codecの実bitstream評価が再現範囲内
- OCR normalizationとCER/WER unit testが合格
- train teacherとheld-out evaluatorが分離
- oracle utilityが同一total bppのtext-mask allocationを上回る

最後の条件が成立しない場合、utility predictorを大規模学習する根拠はありません。

## 配布物の整合性

`MANIFEST.sha256` は同梱ファイルのdigest、`RELEASE_INFO.json` はsnapshotの件数と検証境界、`TREE.txt` はファイル一覧です。再生成は次で行います。

```bash
python scripts/build_release.py --clean-caches --output ../oscar_lic_research_repo.zip
```
