# OCR Text-Crop Actual-Bitstream Smoke v1

**Date:** 2026-06-25 JST  
**Split:** `data/splits/text_crops_parseq_smoke_v1.jsonl`  
**Samples:** 24 latin/mixed word crops from ICDAR2013, ICDAR2015 and TextOCR  
**Purpose:** lock the first OSCAR-LIC OCR-teacher smoke point and verify that codec reconstructions can be evaluated with complete stream-byte accounting.

## Frozen OCR Teacher

- Recognizer: PARSeq base from pinned `external/parseq`
- Repo commit: `1902db043c029a7e03a3818c616c06600af574be`
- Checkpoint: `/root/.cache/torch/hub/checkpoints/parseq-bb5792a6.pt`
- SHA256: `bb5792a68e367476abca029cbf8699abc805f3d3dc7e57aae45c8ec4f7b7cd00`
- Size: `95392675` bytes
- Decode: `decode_ar=true`, `refine_iters=1`
- Preprocess: PARSeq `SceneTextDataModule.get_transform([32, 128])`
- OCR venv lock: `environment/locks.ocr-parseq-py310-cu118.txt`
- OCR env snapshot: `environment/resolved_environment.ocr_parseq_py310_cu118.json`

## Runs

| Run | Result dir | wandb offline run |
|---|---|---|
| Original word crops | `experiments/results/parseq_text_crops_smoke_v1_2026_06_25` | `ukoe0ykv` |
| CompressAI q5 reconstruction OCR | `experiments/results/parseq_text_crops_smoke_v1_compressai_q5_recon_2026_06_25` | `2e3v8wm9` |
| MLIC++ corrected reconstruction OCR | `experiments/results/parseq_text_crops_smoke_v1_mlicpp_recon_2026_06_25` | `169p2fht` |
| Combined summary | local wandb summary only | `pfjj8r9y` |

## Aggregate Results

All bpp values are actual complete file bytes from `compress/decompress` smoke streams, not estimated likelihoods. MS-SSIM is intentionally `null` for this crop split because the 32 px crop height violates the multi-scale downsampling requirement.

| Method | Mean actual bpp | Mean PSNR | unicode CER | unicode exact | latin-alnum CER | latin-alnum exact | Mean PARSeq confidence |
|---|---:|---:|---:|---:|---:|---:|---:|
| Original crop upper bound | n/a | n/a | 0.0714285714 | 0.7083333333 | 0.0359281437 | 0.7916666667 | 0.8650103000 |
| CompressAI hyperprior q5 | 1.4374058418 | 33.3313662807 | 0.0892857143 | 0.7083333333 | 0.0538922156 | 0.7916666667 | 0.8395529824 |
| MLIC++ corrected lambda 0.0250 | 2.7365401728 | 35.1379361749 | 0.0773809524 | 0.7083333333 | 0.0419161677 | 0.7916666667 | 0.8516781020 |

## Held-Out Tesseract Smoke

Tesseract is not used for training, oracle labels or model selection. It is a first held-out recognizer only; its crop performance is much weaker than PARSeq, so a second held-out OCR family is still required before paper claims.

- Package: `tesseract-ocr=5.5.0-1build1`, `libtesseract5=5.5.0-1build1`, `libleptonica6=1.86.0-1`
- Language data: `/usr/share/tesseract-ocr/5/tessdata/eng.traineddata`
- Language data SHA256: `7d4322bd2a7749724879683fc3912cb542f19906c83bcc1a52132556427170b2`
- Fixed word-crop mode: `lang=eng`, `psm=8`, `oem=1`, default DAWG enabled

| Method | unicode CER | unicode exact | latin-alnum CER | latin-alnum exact | wandb offline run |
|---|---:|---:|---:|---:|---|
| Original crop upper bound | 0.3511904762 | 0.2916666667 | 0.2095808383 | 0.5000000000 | `s76l9f39` |
| CompressAI hyperprior q5 | 0.3928571429 | 0.2500000000 | 0.2095808383 | 0.5000000000 | `n6zrsj1n` |
| MLIC++ corrected lambda 0.0250 | 0.3690476190 | 0.1666666667 | 0.2215568862 | 0.4166666667 | `cys9malo` |

## Regression Hashes

- Split `text_crops_parseq_smoke_v1.jsonl`: `49f341b4c6cb00722c01d222fbd5a555267057c124ae259cce0d423f69b7df9d`
- PARSeq original `results.jsonl`: `72c7ad3a7356fee87cea3d9374d8efeaa614e861be257ae727a104f7fbe6d2cd`
- PARSeq original `summary.json`: `649a8e6f1efb4299ff7a52425d49d5998aa119f62f239153244ffbe2439d080a`
- CompressAI q5 codec `results.jsonl`: `d056a44b99baa0ade7d71544e7e0dec9f450e4aa995116ff3db7d56ea605bfea`
- CompressAI q5 codec `summary.json`: `ab6316db6a8c877da9e3eba137fef36b94f902c1c206af8b9681bd01bc86aec4`
- CompressAI q5 OCR `results.jsonl`: `4ef43191aee63ef26eeb65ebe7ec602e52b2d4450486c9d534c5be89efe651a0`
- CompressAI q5 OCR `summary.json`: `33f16533f672db2aa7354c9212f05a2265b4a218a00852e0fff04fb303f33e0c`
- MLIC++ codec `results.jsonl`: `eff764ddd8cdfa1428e785b2cbecf417690c5fa4212c62887630d4e27aada6d2`
- MLIC++ codec `summary.json`: `0ca25dbd6b123a6271c0669ed1b7ae0b2f484528ed19690e5015065461c3b7db`
- MLIC++ OCR `results.jsonl`: `3bb0ed0e704685dcc2386c212c6ba5b52e947d20a78c193f45ddbbc723785157`
- MLIC++ OCR `summary.json`: `f799b52adf66304c52ed55be69ae9413b9f359d8ab66d27089dd06d783f611c3`
- Tesseract PSM8 original `results.jsonl`: `b55f1d27cddf761ab882f30fa4bd322cac8bbf35bb79effe4af4c4dca83f884f`
- Tesseract PSM8 original `summary.json`: `1b9c5b968b782d145238ad88cea230cf493121b77283c6f2d4959ebee661da27`
- Tesseract PSM8 CompressAI q5 OCR `results.jsonl`: `a68ecf76a7cacdf4a1085110321f80169b60b2fded38525fdd0a8febe56b72fb`
- Tesseract PSM8 CompressAI q5 OCR `summary.json`: `a5d9a1f2d012248feb93b19c8289aeed44976653c6d352bd44a41e37b62e1c98`
- Tesseract PSM8 MLIC++ OCR `results.jsonl`: `0074404e5dfd757baaad469b3f8d7777baae199d3ac0bd505c02ada0151cd64b`
- Tesseract PSM8 MLIC++ OCR `summary.json`: `b2eb5c182981326137d240ca7bee93f0741c2f9bfba7671b2255b66a44177d76`

## Interpretation

This is a smoke/regression set, not a publication benchmark. Its value is that it closes the loop from actual codec bitstreams to OCR utility metrics.

The original PARSeq upper bound is imperfect: 17/24 exact under `unicode_strict_v1` and 19/24 exact under `latin_alnum_ci_v1`. The main unstable samples are `62-03` and `fusionopolis`; both are already misrecognized before compression and should be stratified separately in later oracle experiments.

MLIC++ is stronger than CompressAI q5 on this small crop smoke in PSNR and OCR CER, but it spends substantially more actual bpp because small images expose header and entropy-stream overhead. This confirms that OSCAR-LIC should optimize OCR utility per actual serialized bit, not only text-region distortion.

Held-out Tesseract behaves differently from PARSeq and is more brittle to small reconstruction artifacts. This is useful as a reviewer-risk signal: OSCAR-LIC must improve held-out OCR and not only the differentiable teacher.

## Next Decisions

1. Add an original-recognizable subset flag before training any OCR utility model.
2. Add a stronger held-out recognizer family, preferably PaddleOCR or TrOCR, before using held-out results as a paper claim.
3. Run a low-rate crop sweep with matched total bytes, then test whether a simple OCR-utility oracle beats random, confidence, and uniform text allocation controls.
