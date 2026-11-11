# Data

The competition data is private to the ETH Zurich course and is **not** part of this repository (everything under
`data/` except this file is ignored by git). Three files are needed, from the Kaggle competition
`ethz-aml-project-2`:

| file | content | rows | sha256 |
|---|---|---|---|
| `train.csv` | `id`, label `y` (0 to 3), then `x0 ... x17806`: one ECG recording per row at 300 Hz, right-padded with empty cells | 5117 | `d501e869a1ae517b8311e2d0236ee1b0ee15a34bde298c612ff3988b324336d3` |
| `test.csv` | `id`, `x0 ... x17806` | 3411 | `c23377c519109e023ffc0dafa4f637040f2e7eb7a9020ecdc8d3c80d7e943a9f` |
| `sample_submission.csv` | `id`, `y` | 3411 | `cbba780b235212592541f5f3fef2c191a3a1ef41c05ca66028ea2b9bf17ba5fc` |

Recordings are between 2459 and 17 807 samples long (8 to 59 s; median 28 s). Class counts in the training set:
3030 / 443 / 1474 / 170.

## Getting the files

Either copy them from a local download:

```
python scripts/download_data.py --from-local <folder containing the three files>
```

or let the Kaggle API fetch them. The API needs a token that stays inside this repository folder (nothing is written
to your user profile): on kaggle.com open *Settings*, *API*, *Create New Token*, and save the token as
`.kaggle/access_token` (one line) or the classic `.kaggle/kaggle.json`. Then:

```
python scripts/download_data.py --kaggle
```

Both paths verify the hashes above.

## Derived files

- `data/processed/signals.npz`: the recordings as one flat float32 array with offsets, written on the first
  `load_raw()` call (about 70 s); later loads take half a second.
- `data/processed/legacy_2024/`: the two feature tables (`features_train.csv`, `features_test.csv`, 113 columns)
  that the team extracted on 28 November 2024 and that the selected submission was trained on. They are
  reproduction checkpoint A and can be copied in with `--legacy-features <folder>`.
- `data/processed/features_<hash>/`: feature tables and mean heartbeats written by `scripts/extract_features.py`
  for a given configuration.
