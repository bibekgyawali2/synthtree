# Paper draft

Build (Tectonic fetches the LaTeX packages on first use):

```bash
python scripts/paper/make_assets.py --results <results.csv> [<more.csv> ...] --out paper/generated
cd paper && tectonic -X compile main.tex     # -> paper/main.pdf
```

Every number, table and figure in `main.tex` comes from `paper/generated/` (written by
`make_assets.py` from the dataset in `data/output` and the benchmark CSVs), so re-running the two
commands after new results updates the whole paper. Red `[TODO: ...]` marks text that depends on
the real-sensor capture or on the final benchmark. Verify every entry of `refs.bib` against the
publisher record before submission.
