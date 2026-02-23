# Harden: Resumable HF Downloads

**Date**: 2026-02-24
**Branch**: `feat-voxtral-download/supervisor`
**Follows**: `lore/20260224-feat-resumable-hf-downloads.md`

## Assessment

4 opportunities identified after reviewing commit `0a96b74`:

| # | Opportunity | Effort | Impact |
|---|-------------|--------|--------|
| 1 | Duplicated retry-backoff boilerplate (3x) | Quick | Medium |
| 2 | Progress bar ignores completed files on resume (bug) | Quick | High |
| 3 | Extract per-file download into helper function | Easy | Medium |
| 4 | Add cache scanner test for `.partial` exclusion | Easy | Medium |

## Key Finding

**Progress bug (#2)**: `total_downloaded` sums `.partial` sizes but not already-completed `dest` files. On resume with some files done, progress underreports and never reaches 100%.
