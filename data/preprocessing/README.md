# Preprocessing implementations

All nontrivial preprocessing belongs here as versioned code/config, not as undocumented notebook steps. It must preserve raw inputs, consume immutable manifests, record output hashes and distinguish training augmentation from evaluation transforms.

Evaluation resizing, color conversion, crop geometry and OCR aspect-ratio padding are protocol-defining changes and require a decision-log entry.
