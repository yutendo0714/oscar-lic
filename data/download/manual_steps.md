# Manual Acquisition Checklist

For each dataset:

- [ ] open the official source listed in `data/registry.yaml`
- [ ] capture terms/license URL and date
- [ ] record institutional/user acceptance if required
- [ ] verify redistribution restrictions
- [ ] download without modifying filenames where possible
- [ ] generate SHA256 manifest
- [ ] run duplicate audit
- [ ] create split files
- [ ] update registry status and `STATUS.md`

Blocked datasets such as unresolved WebUI/SCI1K sources must not be replaced by ad-hoc web scraping.
