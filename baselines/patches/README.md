# Baseline patches

Store minimal, reviewable patches against pinned external commits here. Each patch requires a companion note with:

- upstream repository and exact commit;
- reason the patch is necessary;
- whether it changes numerical behavior, only paths/interfaces, or only instrumentation;
- commands used to apply/revert it;
- license/provenance statement;
- regression evidence before and after.

Do not silently edit files under `external/` and then report an upstream result.
