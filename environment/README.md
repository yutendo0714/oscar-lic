# Environment Strategy

One monolithic environment is discouraged.

- `mlic_reproduction.yml`: faithful legacy base reproduction
- `ocr_parseq.yml`: OCR teachers/evaluators
- `research_joint.template.yml`: differentiable joint training after compatibility test
- `analysis.yml`: metrics/report generation

After each environment is successfully built, export an explicit lock and store its SHA256. Container digest and NVIDIA driver remain part of the run manifest.
