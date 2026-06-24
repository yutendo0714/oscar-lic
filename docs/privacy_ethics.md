# Privacy, Ethics and Sensitive Text

Text-aware coding can make sensitive information easier to recover. Datasets may contain names, addresses, receipts, screens, account identifiers or medical/financial content.

## Required practices

- minimize storage of OCR strings in logs
- hash or redact identifiers in debugging reports
- restrict raw dataset access
- avoid publishing readable sensitive examples
- do not send dataset images to external OCR APIs without authorization
- evaluate whether gate/text latent increases unauthorized extraction
- keep Track A free of plaintext

## Privacy evaluation

Train an attacker to predict text or sensitive field presence from:

1. file size and section lengths
2. gate/index stream
3. text latent with/without base stream
4. full bitstream without decoder weights

Report utility/privacy Pareto results where relevant. Encryption is the appropriate control for confidentiality; codec obfuscation is not security.
