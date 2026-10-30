# WildEpiPred

Prediction of DNA 5mC sites in two wild grasses, *Oryza rufipogon* and *Aegilops tauschii*.
CG, CHG and CHH methylation are treated as one 5mC class. Each species has its own model.

## Install
The model files are stored with [Git LFS](https://git-lfs.com); install it before cloning.
```
git lfs install
git clone https://github.com/diprosinha/WildEpiPred.git
cd WildEpiPred
conda env create -f environment.yml
conda activate wildepipred
```
Runs on CPU, Apple GPU (MPS) or CUDA. All models are included (`models/`, about 1.3 GB); no internet connection is needed.

## Command line
```
python wildepipred.py predict --species rufipogon --input example/Oryza_rufipogon_region.fasta --out region_predictions.csv
python wildepipred.py predict --species tauschii  --input example/Aegilops_tauschii_sites.fasta --out site_predictions.csv
```
| Option | |
|---|---|
| `--species` | `rufipogon` or `tauschii` |
| `--input` | FASTA, or CSV with a `sequence` column (optional `seq_id`) |
| `--mode` | `site`: each sequence is a 41-bp window with the candidate C at position 21. `scan`: every C on both strands of longer sequences is scored (20 bp flank needed on each side). `auto` (default): `site` if all sequences are 41 bp with C at position 21, otherwise `scan` |
| `--threshold` | probability cut-off for calling 5mC (default 0.5) |

Output CSV columns: `seq_id`, `position` (1-based on the input sequence; for `-` strand cytosines this is the position of the G), `strand`, `context` (CG/CHG/CHH),
`window` (41 bp, read 5'→3' on the cytosine's strand), the probability from each model, `prob_5mC` (ensemble) and `prediction`.

## Web server
Multi-page web interface (Home, Predict, Resources, Developers). Users paste or upload FASTA, choose the
*O. rufipogon*, *Ae. tauschii* or both models, and get a results table, a probability track and sequence view
(scan mode) and a CSV download.

Run directly:
```
python wildepipred.py serve                               # http://127.0.0.1:8501 (this computer only)
python wildepipred.py serve --host 0.0.0.0 --port 8080    # accept outside connections
```
or with Docker (CPU image, any host or cloud platform):
```
docker build -t wildepipred .
docker run -p 8080:8080 wildepipred                       # http://<server>:8080
```

| Setting | Default | |
|---|---|---|
| `--host` / `HOST` | `127.0.0.1` (`0.0.0.0` in Docker) | listening address |
| `--port` / `PORT` | `8501` (`8080` in Docker) | listening port |
| `WILDEPIPRED_MAX_BP` | `20000` | maximum bp per web job; larger inputs should use the command line |

Endpoints: `GET /` (site), `GET /downloads/...` (example data and datasets), `GET /health`,
`POST /predict` (JSON: `species` = `rufipogon` / `tauschii` / `both`, `sequences` = FASTA text, optional `mode`, `threshold`).
The server handles one prediction at a time and has no TLS; for public hosting, put it behind a reverse proxy
(e.g. nginx or Caddy) that provides HTTPS. The Developers page is in `web/index.html`.

## Authors
Dipro Sinha, Subham Ghosh, Abhik Sarkar, Sougata Bhattacharjee, Sayanti Guha Majumdar, Sneha Murmu

## Models
Per species, the ensemble combines five models, with weights chosen to minimise log-loss on the validation set:

| Model | Input |
|---|---|
| Random Forest, XGBoost, LightGBM | Features selected (mutual information → stepwise regression) from NCP, GC content, binary encoding, oligonucleotide frequency, PSTNPss, EIIP and DNABERT-6 embeddings |
| DNABERT | DNABERT-6 fine-tuned on the species' training data |
| DNABERT-BiLSTM | BiLSTM over per-nucleotide encodings + BiLSTM over DNABERT token embeddings |

Ensemble weights (`models/<species>/ensemble_weights.json`): O. rufipogon: DNABERT 0.68, BiLSTM 0.23, LightGBM 0.10;
Ae. tauschii: DNABERT 0.62, BiLSTM 0.28, XGBoost 0.10. Random Forest received weight 0 in both.

## Performance (independent, cluster-disjoint test set)
| Species | Model | ACC | Sn | Sp | MCC | AUROC | AUPRC |
|---|---|---|---|---|---|---|---|
| O. rufipogon | Random Forest | 0.804 | 0.830 | 0.779 | 0.609 | 0.887 | 0.883 |
| | XGBoost | 0.816 | 0.824 | 0.807 | 0.632 | 0.899 | 0.899 |
| | LightGBM | 0.815 | 0.825 | 0.806 | 0.630 | 0.899 | 0.899 |
| | DNABERT | 0.833 | 0.813 | 0.852 | 0.666 | 0.918 | 0.916 |
| | DNABERT-BiLSTM | 0.818 | 0.799 | 0.838 | 0.637 | 0.906 | 0.904 |
| | **Ensemble** | **0.835** | 0.819 | 0.852 | **0.671** | **0.920** | **0.918** |
| Ae. tauschii | Random Forest | 0.713 | 0.710 | 0.717 | 0.426 | 0.792 | 0.803 |
| | XGBoost | 0.722 | 0.704 | 0.740 | 0.445 | 0.805 | 0.816 |
| | LightGBM | 0.723 | 0.706 | 0.740 | 0.446 | 0.805 | 0.816 |
| | DNABERT | 0.745 | 0.737 | 0.754 | 0.491 | 0.836 | 0.843 |
| | DNABERT-BiLSTM | 0.737 | 0.727 | 0.748 | 0.474 | 0.821 | 0.831 |
| | **Ensemble** | **0.750** | 0.740 | 0.760 | **0.500** | **0.840** | **0.848** |

Test sets: O. rufipogon 9,868 + 9,868, Ae. tauschii 8,041 + 8,041 (5mC + negative). Per-context results: `models/<species>/test_metrics.csv`.

## Training data
- Positives: experimental 5mC sites from bisulfite sequencing (2 samples per species; binomial p < 0.01, methylation level ≥ 0.5, coverage ≥ 5).
- Negatives: 41-bp CDS windows with a C at the centre and a CHG motif, < 60% identity, no methylated read at that C.
- Split: CD-HIT-EST 80% cluster-disjoint train/test; no test sequence ≥ 80% identical to a training sequence.

## Limitations
- Negatives come from coding sequences, so predictions partly reflect coding vs non-coding sequence composition (GC content alone gives AUROC 0.81 in O. rufipogon). Scores are most reliable for genomic sequence of the matching species; unusual sequences (e.g. low-complexity repeats) can receive high scores.
- Predictions are not experimental measurements and must not be used as methylation labels.
- No 6mA prediction: no experimental 6mA data exist for these species.
