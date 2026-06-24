# Data acquisition procedures

Automated download is allowed only for entries whose `data/registry.yaml` status and local approval permit it. For manual/registered datasets, store a non-secret acquisition receipt containing source URL, date, release/version and accepted terms; never store credentials or access tokens.

After acquisition, build an immutable JSONL image manifest and verify it with `scripts/validate_dataset_manifest.py`.
