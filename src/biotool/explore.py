"""Group similar sounds by clustering embeddings - mainly for windows no model could identify.

Pipeline: load the embeddings saved by a run -> choose windows (all / unidentified / no
detection) -> L2-normalise -> PCA -> HDBSCAN (finds the number of groups itself and leaves
one-off sounds as 'unassigned') -> 2-D map for display (t-SNE for small sets, else PCA).
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from .models import DETECTORS

SCOPES = {
    "unidentified": "Windows with no species identified (BirdNET/Perch)",
    "nodetection": "Windows with no detection by any model",
    "all": "All windows",
}
MAX_WINDOWS = 20000


def available_models(run_dir: str | Path) -> list[str]:
    files = Path(run_dir, "embeddings").glob("*.npz")
    return sorted({f.stem.rsplit("__", 1)[-1] for f in files if "__" in f.stem})


def _is_species(df: pd.DataFrame) -> pd.Series:
    return (df["model"].isin(["birdnet", "perch_v2", "dfs"]) & (df["scientific_name"].astype(str) != "")
            & ~df["label_key"].astype(str).str.contains(":"))


def load_windows(run_dir: str | Path, model: str) -> tuple[pd.DataFrame, np.ndarray]:
    """One row per embedding window with what the models said about it."""
    run_dir = Path(run_dir)
    win = DETECTORS[model].window_s if model in DETECTORS else 3.0
    metas, vecs = [], []
    for f in sorted((run_dir / "embeddings").glob(f"*__{model}.npz")):
        stem = f.stem.rsplit("__", 1)[0]
        z = np.load(f, allow_pickle=False)
        t = z["start_s"].astype(float)
        src = str(z["source_file"]) if "source_file" in z.files else ""
        metas.append(pd.DataFrame({"stem": stem, "source_file": src, "start_s": t, "end_s": t + win}))
        vecs.append(z["embeddings"].astype(np.float32))
    if not metas:
        raise FileNotFoundError(f"No '{model}' embeddings in {run_dir / 'embeddings'} - run with "
                                f"'Save embeddings' ticked.")
    meta = pd.concat(metas, ignore_index=True)
    X = np.concatenate(vecs)

    labels, species, anydet = [], [], []
    for stem, idx in meta.groupby("stem").groups.items():
        p = run_dir / "per_file" / f"{stem}.detections.csv"
        det = pd.read_csv(p, keep_default_na=False) if p.exists() else pd.DataFrame()
        if not det.empty:
            for c in ("start_s", "end_s", "score"):
                det[c] = pd.to_numeric(det[c])
            det["sp"] = _is_species(det)
            det["name"] = det["common_name"].where(det["common_name"].astype(str) != "", det["label"])
        for i in idx:
            s, e = meta.at[i, "start_s"], meta.at[i, "end_s"]
            d = det[(det["start_s"] < e) & (det["end_s"] > s)] if not det.empty else det
            if d.empty:
                labels.append(""); species.append(False); anydet.append(False)
                continue
            top = d.sort_values("score", ascending=False).drop_duplicates("name").head(3)
            labels.append("; ".join(f"{n} ({sc:.2f})" for n, sc in zip(top["name"], top["score"])))
            species.append(bool(d["sp"].any())); anydet.append(True)
    meta["labels"] = labels
    meta["identified"] = species
    meta["any_detection"] = anydet
    return meta, X


def cluster(meta: pd.DataFrame, X: np.ndarray, scope: str = "unidentified", min_cluster_size: int = 8,
            seed: int = 0, progress=None) -> tuple[pd.DataFrame, pd.DataFrame]:
    from sklearn.cluster import HDBSCAN
    from sklearn.decomposition import PCA

    sel = {"unidentified": ~meta["identified"], "nodetection": ~meta["any_detection"],
           "all": pd.Series(True, index=meta.index)}[scope].to_numpy()
    m, Xs = meta[sel].reset_index(drop=True), X[sel]
    note = ""
    if len(m) > MAX_WINDOWS:
        rng = np.random.default_rng(seed)
        keep = np.sort(rng.choice(len(m), MAX_WINDOWS, replace=False))
        m, Xs = m.iloc[keep].reset_index(drop=True), Xs[keep]
        note = f" (random sample of {MAX_WINDOWS})"
    if len(m) < 3:
        raise ValueError(f"Only {len(m)} windows in this selection - nothing to group.")
    if progress:
        progress(f"Clustering {len(m)} windows{note} ...")
    Xn = Xs / (np.linalg.norm(Xs, axis=1, keepdims=True) + 1e-9)
    k = min(32, Xn.shape[1], len(m) - 1)
    Z = PCA(n_components=k, random_state=seed).fit_transform(Xn)
    mcs = max(2, min(min_cluster_size, len(m) // 2))
    lab = HDBSCAN(min_cluster_size=mcs, min_samples=max(1, mcs // 2), copy=True).fit_predict(Z)
    m["cluster"] = lab

    if len(m) <= 4000:
        from sklearn.manifold import TSNE

        if progress:
            progress("Computing 2-D map (t-SNE) ...")
        xy = TSNE(n_components=2, perplexity=min(30, max(5, len(m) // 10)), init="pca",
                  random_state=seed).fit_transform(Z)
    else:
        xy = Z[:, :2]
    m["x"], m["y"] = xy[:, 0], xy[:, 1]

    # representative window (closest to the cluster mean) + summary
    m["representative"] = False
    rows = []
    for c, d in m.groupby("cluster"):
        idx = d.index.to_numpy()
        centre = Z[idx].mean(axis=0)
        rep = idx[np.argmin(np.linalg.norm(Z[idx] - centre, axis=1))]
        if c >= 0:
            m.at[rep, "representative"] = True
        lab_counts = (d["labels"].str.split("; ").explode().str.replace(r" \(\d\.\d+\)$", "", regex=True))
        lab_counts = lab_counts[lab_counts != ""].value_counts()
        rows.append(dict(
            cluster=int(c), name="" if c >= 0 else "(unassigned)", n_windows=len(d),
            n_files=d["stem"].nunique(),
            share_with_labels=round(float((d["labels"] != "").mean()), 2),
            frequent_labels=", ".join(f"{k} ({v})" for k, v in lab_counts.head(3).items()),
            first_file=d.iloc[0]["stem"],
        ))
    summary = pd.DataFrame(rows).sort_values(["cluster"], key=lambda s: s.where(s >= 0, 10**9))
    if progress:
        progress(f"Found {int((summary['cluster'] >= 0).sum())} sound groups; "
                 f"{int((m['cluster'] < 0).sum())} windows left unassigned.")
    return m, summary.reset_index(drop=True)


def export(run_dir: str | Path, windows: pd.DataFrame, summary: pd.DataFrame, model: str) -> Path:
    names = dict(zip(summary["cluster"], summary["name"]))
    out = windows.assign(cluster_name=windows["cluster"].map(names), embedding_model=model)
    cols = ["stem", "source_file", "start_s", "end_s", "cluster", "cluster_name", "labels",
            "identified", "embedding_model"]
    p = Path(run_dir) / f"sound_groups_{model}.csv"
    out[cols].to_csv(p, index=False)
    summary.to_csv(Path(run_dir) / f"sound_groups_{model}_summary.csv", index=False)
    return p
