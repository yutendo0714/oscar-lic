# Configuration Policy

All reported runs use a committed YAML config. Parent/override semantics are descriptive until the training launcher is implemented; the final launcher must resolve the full config and save the resolved YAML plus SHA256 in the run directory.

Config changes that affect data, rate accounting, OCR normalization, primary metrics or fairness require a new experiment ID and, where substantive, an ADR.

No secret, credential, absolute personal path or dataset byte is stored in config. Machine-local paths belong in an ignored local override whose resolved values are captured in the run manifest.
