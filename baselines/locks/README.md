# Resolved external locks

Generated lock files belong here after a repository, checkpoint or model bundle is actually resolved. Never hand-edit a resolved hash to make a run appear reproducible.

Expected records include:

- `resolved_repositories.json`: requested URL, registered pin, resolved HEAD, submodule state and checkout time;
- `resolved_checkpoints.json`: source, byte length, SHA-256, loader format and architecture compatibility;
- `resolved_ocr_models.json`: code pin, model revision, weight/tokenizer/dictionary hashes and decoding profile.

A branch or model alias alone is not a lock.
