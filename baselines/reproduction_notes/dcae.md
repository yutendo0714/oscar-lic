# DCAE Reproduction Note

- Repository: `CVL-UESTC/DCAE`
- Commit: `e2525a00467cbc326045674c7e5e0f1d9964604b`
- Official README describes OpenImages training and Kodak/CLIC/TESTIMAGES evaluation.
- Actual file compression/decompression entry: `compress_and_decompress.py`.

## Important cautions

- The README's historical clone command points to a different owner path than the inspected repository; use the registered URL and pin.
- Dependencies are listed without versions (`compressai`, `tensorboard`, `thop`, `timm`). Build and freeze a working lock rather than installing into another experiment environment.
- Verify repository license before copying code; README presence is not a license grant.
- Use one checkpoint only after recording its source and SHA256.
- Confirm that evaluation and actual stream scripts use the same model path and preprocessing.

DCAE is a secondary base and dictionary-prior ablation. Do not let it delay G1 on MLIC++.
