#!/usr/bin/env python3
"""WiMeP: prediction of DNA 5mC sites in Oryza rufipogon and Aegilops tauschii.

Each species has its own model: an ensemble of Random Forest, XGBoost, LightGBM (on selected handcrafted +
DNABERT features), fine-tuned DNABERT-6 and a DNABERT-BiLSTM. 5mC = CG, CHG and CHH methylation as one class.

Input: FASTA (or CSV with a `sequence` column).
  site mode  every sequence is a 41-bp window with the candidate C at position 21
  scan mode  sequences of >= 41 bp are cut into overlapping 41-bp windows (step 1 bp); every window centred on a
             cytosine (C, or G = C on the opposite strand) is scored
  auto       site mode if all sequences are 41 bp with C at position 21, otherwise scan mode

usage:
  python wimep.py predict --species rufipogon --input seqs.fasta --out predictions.csv
  python wimep.py serve   [--host 0.0.0.0] [--port 8501]   # web server (http://127.0.0.1:8501)
"""
import argparse, functools, io, json, os, re, sys
import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import features as F  # noqa: E402

SPECIES = {"rufipogon": "Oryza_rufipogon", "tauschii": "Aegilops_tauschii"}
BASE_MODEL = os.path.join(HERE, "models", "dnabert_base")  # frozen zhihan1996/DNA_bert_6 (embeddings)
MODELS = ["RF", "XGBoost", "LightGBM", "DNABERT", "DNABERT-BiLSTM"]
HALF, COMP = 20, str.maketrans("ACGTN", "TGCAN")
MAX_WEB_BP = int(os.environ.get("WIMEP_MAX_BP", 20000))


def kmers(s, k=6):
    return " ".join(s[i:i + k] for i in range(len(s) - k + 1))


def context(w):
    return "CG" if w[HALF + 1] == "G" else "CHG" if w[HALF + 2] == "G" else "CHH"


# ---------------------------------------------------------------- input
def read_input(text_or_path):
    """Return DataFrame(seq_id, sequence) from a FASTA/CSV path or raw FASTA / plain-sequence text."""
    if os.path.exists(text_or_path):
        if text_or_path.endswith((".csv", ".csv.gz")):
            df = pd.read_csv(text_or_path, dtype=str)
            if "seq_id" not in df: df.insert(0, "seq_id", [f"seq{i + 1}" for i in range(len(df))])
            return df[["seq_id", "sequence"]]
        text = open(text_or_path).read()
    else:
        text = text_or_path
    ids, seqs = [], []
    for line in text.splitlines():
        line = line.strip()
        if not line: continue
        if line.startswith(">"): ids.append(line[1:].split()[0] or f"seq{len(ids) + 1}"); seqs.append("")
        else:
            if not ids: ids.append("seq1"); seqs.append("")
            seqs[-1] += re.sub(r"\s", "", line)
    return pd.DataFrame(dict(seq_id=ids, sequence=[s.upper().replace("U", "T") for s in seqs]))


def windows(df, mode="auto"):
    """41-bp windows centred on a cytosine.

    site mode: each sequence is one 41-bp window with C at position 21.
    scan mode: the sequence (>= 41 bp) is cut into overlapping 41-bp windows with a step of 1 bp; every window whose
    centre is C (cytosine on the given strand) or G (cytosine on the opposite strand, window reverse-complemented)
    is scored. Columns: seq_id, position (1-based centre), strand, context, window_start, window_end, window."""
    is_site = (df.sequence.str.len() == 41) & (df.sequence.str[HALF] == "C")
    if mode == "auto": mode = "site" if is_site.all() else "scan"
    rows = []
    if mode == "site":
        for r in df.itertuples():
            if len(r.sequence) != 41 or r.sequence[HALF] != "C":
                raise ValueError(f"{r.seq_id}: each sequence must be 41 bp with C at position 21 (got {len(r.sequence)} bp)")
            if re.fullmatch("[ACGT]+", r.sequence):
                rows.append((r.seq_id, 21, "+", context(r.sequence), 1, 41, r.sequence))
    else:
        for r in df.itertuples():
            s = r.sequence
            for i in range(HALF, len(s) - HALF):
                if s[i] not in "CG": continue
                w = s[i - HALF:i + HALF + 1]
                if s[i] == "G": w = w.translate(COMP)[::-1]
                if re.fullmatch("[ACGT]+", w):
                    rows.append((r.seq_id, i + 1, "+" if s[i] == "C" else "-", context(w), i - HALF + 1, i + HALF + 1, w))
    return pd.DataFrame(rows, columns=["seq_id", "position", "strand", "context", "window_start", "window_end", "window"]), mode


# ---------------------------------------------------------------- models
@functools.lru_cache(None)
def device():
    import torch
    return "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"


@functools.lru_cache(None)
def base_bert():
    from transformers import BertModel, BertTokenizer
    return BertTokenizer.from_pretrained(BASE_MODEL), BertModel.from_pretrained(BASE_MODEL).to(device()).eval()


@functools.lru_cache(None)
def load_species(sp):
    import joblib, torch
    from transformers import AutoTokenizer, BertForSequenceClassification
    d = os.path.join(HERE, "models", SPECIES[sp])
    ml = joblib.load(os.path.join(d, "ml_models.joblib"))
    pst = F.PSTNPss(); pst.z = ml["pstnp"]
    tok = AutoTokenizer.from_pretrained(os.path.join(d, "dnabert_model"))
    ft = BertForSequenceClassification.from_pretrained(os.path.join(d, "dnabert_model")).to(device()).eval()
    bil = BiLSTM(); bil.load_state_dict(torch.load(os.path.join(d, "bilstm.pt"), map_location="cpu")); bil.to(device()).eval()
    w = json.load(open(os.path.join(d, "ensemble_weights.json")))
    return dict(ml=ml, pst=pst, tok=tok, ft=ft, bilstm=bil, weights=w)


def BiLSTM(nt_dim=8, bert_dim=64, hidden=64, drop=0.3):
    """DNABERT-BiLSTM architecture (same as training): BiLSTM over nucleotide channels + BiLSTM over DNABERT tokens."""
    import torch, torch.nn as nn

    class Net(nn.Module):
        def __init__(self):
            super().__init__()
            self.lstm_nt = nn.LSTM(nt_dim, hidden, batch_first=True, bidirectional=True)
            self.lstm_bert = nn.LSTM(bert_dim, hidden, batch_first=True, bidirectional=True)
            self.head = nn.Sequential(nn.Dropout(drop), nn.Linear(8 * hidden, 64), nn.ReLU(), nn.Dropout(drop), nn.Linear(64, 1))

        def forward(self, nt, bert):
            a, _ = self.lstm_nt(nt); b, _ = self.lstm_bert(bert)
            return self.head(torch.cat([a.mean(1), a.amax(1), b.mean(1), b.amax(1)], 1)).squeeze(1)
    return Net()


def score(seqs, sp, bs=512):
    """Per-model and ensemble 5mC probabilities for a list of 41-bp windows."""
    import joblib, torch
    M = load_species(sp); tok, bert = base_bert(); pca = joblib.load(os.path.join(HERE, "token_pca.joblib"))
    dev = device(); mean, tokp, p_ft = [], [], []
    with torch.no_grad():
        for i in range(0, len(seqs), bs):
            b = seqs[i:i + bs]
            h = bert(**tok([kmers(s) for s in b], return_tensors="pt", padding=True).to(dev)).last_hidden_state[:, 1:-1, :].float().cpu().numpy()
            mean.append(h.mean(1)); tokp.append(pca.transform(h.reshape(-1, 768)).reshape(len(b), 36, -1))
            e = M["tok"]([kmers(s) for s in b], padding="max_length", max_length=38, truncation=True, return_tensors="pt").to(dev)
            p_ft.append(torch.softmax(M["ft"](**e).logits.float(), -1)[:, 1].cpu().numpy())
    # training used float16-stored embeddings; round the same way
    mean = np.concatenate(mean).astype(np.float16).astype(np.float32)
    tokp = np.concatenate(tokp).astype(np.float16).astype(np.float32)
    ml = M["ml"]
    X = pd.concat([F.handcrafted(seqs, M["pst"]), pd.DataFrame(mean, columns=[f"BERT_{i}" for i in range(768)])], axis=1)[ml["all_features"]]
    Z = pd.DataFrame(ml["scaler"].transform(X.values), columns=ml["all_features"])[ml["selected"]].values
    out = pd.DataFrame({m: ml["models"][m].predict_proba(Z)[:, 1] for m in ("RF", "XGBoost", "LightGBM")})
    out["DNABERT"] = np.concatenate(p_ft)
    nt = F.per_position(seqs); pb = []
    with torch.no_grad():
        for i in range(0, len(seqs), 2048):
            pb.append(torch.sigmoid(M["bilstm"](torch.tensor(nt[i:i + 2048], device=dev), torch.tensor(tokp[i:i + 2048], device=dev))).cpu().numpy())
    out["DNABERT-BiLSTM"] = np.concatenate(pb)
    out["prob_5mC"] = sum(M["weights"][m] * out[m] for m in MODELS)
    return out


def predict(df, sp, mode="auto", threshold=0.5, chunk=20000):
    W, mode = windows(df, mode)
    if W.empty: return W.assign(prob_5mC=[], prediction=[]), mode
    parts = [score(W.window.iloc[i:i + chunk].tolist(), sp) for i in range(0, len(W), chunk)]
    P = pd.concat(parts, ignore_index=True)
    W = pd.concat([W, P.astype(np.float64).round(4)], axis=1)
    W["prediction"] = np.where(W.prob_5mC >= threshold, "5mC", "non-5mC")
    return W, mode


# ---------------------------------------------------------------- CLI / web
def serve(host, port):
    """Web server: static site from web/ (pages, images, downloads) and POST /predict (JSON)."""
    import mimetypes, threading
    from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
    from urllib.parse import unquote, urlparse
    root = os.path.realpath(os.path.join(HERE, "web"))
    lock = threading.Lock()  # one prediction at a time; the models are shared

    def run(req):
        sp = req.get("species")
        if sp not in list(SPECIES) + ["both"]: raise ValueError("choose a species model")
        df = read_input(req.get("sequences", "").strip() or ">")
        df = df[df.sequence != ""]
        if df.empty: raise ValueError("no sequence given")
        short = df[df.sequence.str.len() < 41]
        if len(short):
            raise ValueError("each sequence must be at least 41 bp; too short: " +
                             ", ".join(f"{r.seq_id} ({len(r.sequence)} bp)" for r in short.head(5).itertuples()) + (" ..." if len(short) > 5 else ""))
        if df.sequence.str.len().sum() > MAX_WEB_BP:
            raise ValueError(f"web input is limited to {MAX_WEB_BP:,} bp; use the command-line tool for larger inputs")
        out = {}
        for s in (SPECIES if sp == "both" else [sp]):
            with lock: res, mode = predict(df, s, req.get("mode", "auto"), float(req.get("threshold", 0.5)))
            if res.empty: raise ValueError("no window could be scored: no 41-bp window has a cytosine (C, or G on the opposite strand) at its centre with only A/C/G/T around it")
            out[s] = dict(species=SPECIES[s].replace("_", " "), mode=mode, n=len(res), n_5mC=int((res.prediction == "5mC").sum()),
                          columns=list(res.columns), rows=res.values.tolist(), csv=res.to_csv(index=False))
        seqs = {r.seq_id: r.sequence for r in df.itertuples()}
        return dict(results=out, sequences=seqs if sum(map(len, seqs.values())) <= MAX_WEB_BP else {})

    class H(BaseHTTPRequestHandler):
        def _send(self, code, body, ctype, extra=()):
            self.send_response(code); self.send_header("Content-Type", ctype); self.send_header("Content-Length", str(len(body)))
            for k, v in extra: self.send_header(k, v)
            self.end_headers(); self.wfile.write(body)

        def do_GET(self):
            path = unquote(urlparse(self.path).path)
            if path == "/health": return self._send(200, b"ok", "text/plain")
            f = os.path.realpath(os.path.join(root, path.lstrip("/") or "index.html"))
            if os.path.isdir(f): f = os.path.join(f, "index.html")
            if not f.startswith(root + os.sep) or not os.path.isfile(f): return self._send(404, b"not found", "text/plain")
            ctype = mimetypes.guess_type(f)[0] or "application/octet-stream"
            if ctype.startswith("text/") or ctype.endswith(("javascript", "svg+xml", "json")): ctype += "; charset=utf-8"
            extra = [("Content-Disposition", f'attachment; filename="{os.path.basename(f)}"')] if "/downloads/" in f else []
            self._send(200, open(f, "rb").read(), ctype, extra)

        def do_POST(self):
            if urlparse(self.path).path != "/predict": return self._send(404, b"not found", "text/plain")
            try:
                n = int(self.headers.get("Content-Length", 0))
                if n > 4 * MAX_WEB_BP + 100_000: raise ValueError(f"web input is limited to {MAX_WEB_BP:,} bp")
                body = run(json.loads(self.rfile.read(n)))
                self._send(200, json.dumps(body).encode(), "application/json")
            except Exception as e:  # report input errors to the page
                self._send(400, json.dumps(dict(error=str(e))).encode(), "application/json")

        def log_message(self, *a): pass

    print("loading models ...", flush=True)
    for sp in SPECIES: load_species(sp)
    base_bert()
    print(f"WiMeP web server: http://{'127.0.0.1' if host in ('0.0.0.0', '') else host}:{port}  (Ctrl+C to stop)", flush=True)
    ThreadingHTTPServer((host, port), H).serve_forever()


def main():
    ap = argparse.ArgumentParser(description="WiMeP: 5mC site prediction for Oryza rufipogon and Aegilops tauschii")
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("predict", help="score sequences from a FASTA/CSV file")
    p.add_argument("--species", required=True, choices=sorted(SPECIES))
    p.add_argument("--input", required=True, help="FASTA, or CSV with a `sequence` column")
    p.add_argument("--out", required=True, help="output CSV")
    p.add_argument("--mode", default="auto", choices=["auto", "site", "scan"])
    p.add_argument("--threshold", type=float, default=0.5)
    s = sub.add_parser("serve", help="web server")
    s.add_argument("--host", default=os.environ.get("HOST", "127.0.0.1"), help="0.0.0.0 to accept outside connections")
    s.add_argument("--port", type=int, default=int(os.environ.get("PORT", 8501)))
    a = ap.parse_args()
    if a.cmd == "serve": return serve(a.host, a.port)
    if not os.path.exists(a.input): sys.exit(f"input not found: {a.input}")
    df = read_input(a.input); short = (df.sequence.str.len() < 41).sum()
    if short: print(f"warning: {short} sequence(s) shorter than 41 bp skipped", file=sys.stderr)
    res, mode = predict(df[df.sequence.str.len() >= 41], a.species, a.mode, a.threshold)
    res.to_csv(a.out, index=False)
    print(f"{mode} mode: {len(res)} cytosines scored, {(res.prediction == '5mC').sum()} predicted 5mC -> {a.out}")


if __name__ == "__main__":
    main()
