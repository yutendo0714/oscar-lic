# Data Operations

Raw bytes are not bundled. Every acquired dataset needs:

1. approved source and terms
2. immutable raw manifest with relative path, byte size and SHA256
3. split manifest
4. preprocessing version and output manifest
5. duplicate/leakage audit

Expected layout:

```text
data/raw/<dataset>/          # immutable, ignored by git
data/manifests/<dataset>.jsonl
data/splits/<dataset>_{train,val,test}.txt
data/processed/<pipeline_hash>/
```

Never infer permission from a paper using a dataset. Verify the source terms directly.
