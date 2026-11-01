"""DeepForestSound (MNHN / One Forest Vision) - 12 species of African tropical forests.

Dubus et al. (2026) DeepForestSound: A Multi-Species Automatic Detector for Passive Acoustic
Monitoring in African Tropical Forests. arXiv:2604.08087. Code + weights: CC BY-NC-SA 4.0.

Two AST + LoRA models on 10 s windows:
  * LF (1 kHz audio, 3-250 Hz) - African forest elephant rumbles
  * MF (8 kHz audio, 50-4000 Hz) - 11 birds and primates (incl. chimpanzee)

To respect the share-alike licence, BioTool does not include DFS code: it downloads the original
code and weights (pinned to a tested commit, checksums verified) into the models folder on first
use. Needs the optional PyTorch dependencies:  InstallDeepForestSound.bat  /  uv sync --extra dfs
"""

from __future__ import annotations

import hashlib
import importlib.util
import sys
import urllib.request
from pathlib import Path

import numpy as np

from ..audio import frame_signal
from ..paths import models_dir
from .base import ModelAdapter, ModelOutput

COMMIT = "396dccc600c85f648efe77549a86a0612954abac"
RAW = "https://raw.githubusercontent.com/MNHN-OFVI/DeepForestSound/{}/{}"
LFS = "https://media.githubusercontent.com/media/MNHN-OFVI/DeepForestSound/{}/{}"
CODE_FILES = ["src/ast_models.py", "src/LoRA_inject.py", "src/utils.py"]
WEIGHTS = {  # file -> sha256 (from the repository's Git LFS pointers)
    "model_weights/DFS_LF.pth": "ca263f579cd40bf5ee40cca0c318f549c631885dd5afcecb01fc856c0c9b6540",
    "model_weights/DFS_MF.pth": "0c962881e42cf9a2b8b0fe668bc3cffc0ae6b264d486454671ff24d768f842e5",
}

# column order of the models' outputs: LF (1 class) + MF (11 classes)
SPECIES = [
    ("loxodonta", "Loxodonta cyclotis", "African Forest Elephant", 0.30),
    ("balearica_regulorum", "Balearica regulorum", "Grey Crowned Crane", 0.82),
    ("bycanistes_subcylindricus", "Bycanistes subcylindricus", "Black-and-white-casqued Hornbill", 0.33),
    ("chrysococcyx_cupreus", "Chrysococcyx cupreus", "African Emerald Cuckoo", 0.10),
    ("colobus_guereza", "Colobus guereza", "Mantled Guereza", 0.71),
    ("corythaeola_cristata", "Corythaeola cristata", "Great Blue Turaco", 0.44),
    ("laniarius_mufumbiri", "Laniarius mufumbiri", "Papyrus Gonolek", 0.41),
    ("lophocebus_albigena", "Lophocebus albigena", "Grey-cheeked Mangabey", 0.80),
    ("pan_troglodytes", "Pan troglodytes", "Chimpanzee", 0.99),
    ("streptopelia_semitorquata", "Streptopelia semitorquata", "Red-eyed Dove", 0.20),
    ("tauraco_schuettii", "Tauraco schuettii", "Black-billed Turaco", 0.71),
    ("turtur_tympanistria", "Turtur tympanistria", "Tambourine Dove", 0.10),
]
SR_LF, SR_MF, MEL, TL_LF, TL_MF = 1000, 8000, 128, 256, 512


def available() -> tuple[bool, str]:
    """Are the optional PyTorch dependencies installed?"""
    missing = [m for m in ("torch", "torchaudio", "timm", "peft") if importlib.util.find_spec(m) is None]
    if missing:
        return False, f"not installed ({', '.join(missing)} missing) - run InstallDeepForestSound.bat"
    return True, ""


def _sha256(p: Path) -> str:
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _fetch(url: str, dest: Path, progress=None) -> None:
    tmp = dest.with_suffix(dest.suffix + ".part")
    req = urllib.request.Request(url, headers={"User-Agent": "biotool"})
    with urllib.request.urlopen(req, timeout=60) as r, open(tmp, "wb") as f:
        total = int(r.headers.get("Content-Length") or 0)
        done, last = 0, -10
        while chunk := r.read(1 << 20):
            f.write(chunk)
            done += len(chunk)
            if progress and total and (pct := int(100 * done / total)) >= last + 10:
                progress(f"  {dest.name}: {pct}%")
                last = pct
    tmp.replace(dest)


def ensure_files(progress=None) -> Path:
    root = models_dir() / "deepforestsound" / COMMIT[:12]
    for rel in CODE_FILES:
        p = root / rel
        if not p.exists():
            p.parent.mkdir(parents=True, exist_ok=True)
            _fetch(RAW.format(COMMIT, rel), p)
    for rel, sha in WEIGHTS.items():
        p = root / rel
        if p.exists() and p.stat().st_size > 1_000_000:
            continue
        p.parent.mkdir(parents=True, exist_ok=True)
        if progress:
            progress(f"DeepForestSound: downloading {p.name} (~350 MB, once)")
        _fetch(LFS.format(COMMIT, rel), p, progress)
        if _sha256(p) != sha:
            p.unlink()
            raise RuntimeError(f"Checksum mismatch for {p.name} - download corrupted, try again")
    return root


def _load_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


class DeepForestSound(ModelAdapter):
    key = "dfs"
    name = "DeepForestSound (Africa)"
    samplerate = SR_MF
    native_sr = True          # gets the file's own sample rate; resamples like the original code
    window_s = 10.0
    default_threshold = 0.5
    fmax_hz = 4000
    licence_note = "DeepForestSound: CC BY-NC-SA 4.0 (non-commercial). Optional - needs PyTorch."
    description = ("12 species of African tropical forests (Uganda): forest elephant rumbles, chimpanzee, "
                   "2 monkeys, 8 birds. Specialist model - see README before using elsewhere.")
    advanced = [
        dict(key="thresholds", label="Thresholds", kind="choice", default="per species (DFS paper)",
             choices=["per species (DFS paper)", "min score above"],
             help="DFS species need very different thresholds (chimpanzee 0.99, doves 0.10).\n"
                  "'per species' uses the values suggested by the DFS authors; the min score\n"
                  "box is then ignored."),
        dict(key="overlap_s", label="Window overlap (s)", kind="float", default=0.0, range=(0.0, 8.0, 1.0),
             help="Overlap between 10 s windows (DFS default: none)."),
    ] + ModelAdapter.advanced

    @classmethod
    def availability(cls) -> tuple[bool, str]:
        return available()

    def load(self, progress=None) -> None:
        ok, msg = available()
        if not ok:
            raise RuntimeError(f"DeepForestSound is {msg}")
        import torch
        import torchaudio

        if self.threads:
            torch.set_num_threads(self.threads)
        root = ensure_files(progress)
        ast = _load_module("biotool_dfs_ast_models", root / "src" / "ast_models.py")
        lora = _load_module("biotool_dfs_lora", root / "src" / "LoRA_inject.py")
        self._utils = _load_module("biotool_dfs_utils", root / "src" / "utils.py")
        if progress:
            progress("DeepForestSound: loading models ...")
        kw = dict(fstride=10, tstride=10, input_fdim=MEL, imagenet_pretrain=False, audioset_pretrain=False,
                  model_size="base384", verbose=False)
        self._lf = lora.inject_lora(ast.ASTModel(label_dim=1, input_tdim=TL_LF, **kw))
        self._mf = lora.inject_lora(ast.ASTModel(label_dim=len(SPECIES) - 1, input_tdim=TL_MF, **kw))
        for m, rel in ((self._lf, "model_weights/DFS_LF.pth"), (self._mf, "model_weights/DFS_MF.pth")):
            m.load_state_dict(torch.load(root / rel, map_location="cpu", weights_only=True))
            m.eval()
        self._emb = None
        self._mf.mlp_head.register_forward_hook(lambda mod, inp, out: setattr(self, "_emb", inp[0].detach()))
        self._torch, self._ta = torch, torchaudio
        self.input_sr = SR_MF

    def _thresholds(self) -> np.ndarray:
        if str(self.options.get("thresholds", "per species")).startswith("per species"):
            return np.array([s[3] for s in SPECIES])
        return np.full(len(SPECIES), self.threshold)

    def process(self, audio, offset_s, file):
        out = ModelOutput()
        torch, ta = self._torch, self._ta
        sr = int(getattr(self, "input_sr", SR_MF))
        x = torch.from_numpy(np.ascontiguousarray(audio, dtype=np.float32))[None, :]
        x_lf = ta.functional.resample(x, sr, SR_LF)[0].numpy() if sr != SR_LF else x[0].numpy()
        x_mf = ta.functional.resample(x, sr, SR_MF)[0].numpy() if sr != SR_MF else x[0].numpy()
        hop_s = max(1.0, self.window_s - float(self.options.get("overlap_s", 0.0)))
        f_lf = frame_signal(x_lf, int(self.window_s * SR_LF), int(hop_s * SR_LF))
        f_mf = frame_signal(x_mf, int(self.window_s * SR_MF), int(hop_s * SR_MF))
        n = min(len(f_lf), len(f_mf))
        # Like the original DFS code: a last window that would run past the end of the audio is
        # shifted back so it ends exactly at the end (no zero padding).
        starts = [i * hop_s for i in range(n)]
        dur = len(x_mf) / SR_MF
        if n and starts[-1] + self.window_s > dur + 1e-6 and dur >= self.window_s:
            starts[-1] = dur - self.window_s
            f_lf[n - 1] = x_lf[-int(self.window_s * SR_LF):]
            f_mf[n - 1] = x_mf[-int(self.window_s * SR_MF):]
        idx = self.loud_indices(f_mf[:n])
        if len(idx) == 0:
            return out
        thr = self._thresholds()
        u = self._utils
        for b in range(0, len(idx), 8):
            ii = idx[b:b + 8]
            X_lf = torch.stack([u.make_features_lf(f_lf[i], SR_LF, MEL, TL_LF) for i in ii]).float()
            X_mf = torch.stack([u.make_features_mf(f_mf[i], SR_MF, MEL, TL_MF) for i in ii]).float()
            with torch.no_grad():
                y = torch.cat([torch.sigmoid(self._lf(X_lf)), torch.sigmoid(self._mf(X_mf))], dim=1).numpy()
            emb = self._emb.numpy() if self._emb is not None else None
            for k, fi in enumerate(ii):
                start = offset_s + starts[fi]
                hits = [j for j in np.argsort(-y[k]) if y[k, j] >= thr[j]][: self.max_per_window]
                for j in hits:
                    col, sci, common, _ = SPECIES[j]
                    out.detections.append(dict(
                        file=file, start_s=round(start, 3), end_s=round(start + self.window_s, 3),
                        model=self.key, label=sci, label_key=sci.lower(), scientific_name=sci,
                        common_name=common, group="biophony", score=round(float(y[k, j]), 4)))
                if emb is not None:
                    out.emb_times.append(round(start, 3))
                    out.embeddings.append(emb[k].astype(np.float16))
        return out
