#!/usr/bin/env python3
"""DreamSim for every finished run folder of a known format under one or more root folders.

Uses the DreamSim setup of dreamsim_eval.py (next to this file): dreamsim 0.2.1, the package's default ensemble
model and its own 224x224 preprocess, GPU if available. For each run folder it compares the saved
reconstructions with the matching originals and writes dreamsim.json into the run folder (n, mean, std, min,
max and the per-image distances, per pair), in the same format as dreamsim_eval.py. Folders that already have a
dreamsim.json are not recomputed; their stored values go into the table.

Known formats (run folder = the folder holding the marker):
  RQ1 SDAR     <name>/output/<name>_metrics.json  ->  output/<name>_orig.npy vs output/<name>_recon.npy
  RQ1 MAERA    maera/<name>_ep500/summary.json    ->  eval/<dataset>_orig.npy (the fixed originals; the 32x32
               one for _in32 runs) vs eval128_recon.npy. Only folders whose name ends in _ep500.
  RQ1 PCAT     <name>/metrics.json                ->  orig.npy vs recon.npy
  RQ1 UnSplit  <name>/summary.json                ->  originals/*.png vs reconstructions/*.png, paired by name (20)
  RQ3 SDAR     <defense>/<cut>/sdar/full_sdar_summary.json -> originals.npy vs reconstructions.npy (the 1,280
               validation images, pair "eval") and fixed_probe_originals.npy vs fixed_probe_reconstructions.npy
               (the 20-image fixed probe, pair "fixed_probe"), reported separately.

    python dreamsim_all.py sdarresnet18-10 maeraresnet18-100 pcat_rq1 unsplitresnet18-tiny
    python dreamsim_all.py . --quick          # only the first folder still to do; prints its result
"""

import argparse
import json
import re
import sys
from pathlib import Path

import numpy as np

from dreamsim_eval import (DATASET_RE, RESULT_FILE, dataset_eval_file, dreamsim_distances, load_dreamsim,
                           load_png_dirs, rel_path, to_uint8_nhwc, write_json)

ROOT = Path(__file__).resolve().parent
SKIP_DIRS = {".git", "checkpoints", "__pycache__", "data", "datasets"}
ATTACK_DIR_PREFIX = (("sdarresnet18", "sdar"), ("maeraresnet18", "maera"), ("pcat", "pcat"), ("unsplitresnet18", "unsplit"))
FOLDER_DATASET = (("-100", "cifar100"), ("-10", "cifar10"), ("-tiny", "tinyimagenet"), ("_tiny", "tinyimagenet"))
UNSPLIT_N = 20


# ---------------------------------------------------------------------------
# run folders
# ---------------------------------------------------------------------------
def walk_markers(root):
    """Yield (folder, file name) for every marker file under root, skipping checkpoints, data and .git."""
    stack = [Path(root)]
    while stack:
        d = stack.pop()
        try:
            entries = sorted(d.iterdir())
        except OSError:
            continue
        for e in entries:
            if e.is_dir():
                if e.name not in SKIP_DIRS and not e.is_symlink():
                    stack.append(e)
            elif e.name in ("full_sdar_summary.json", "metrics.json", "summary.json") or (
                    e.name.endswith("_metrics.json") and d.name == "output"):
                yield d, e.name


def attack_from_path(run_dir):
    """sdar / maera / pcat / unsplit from the attack folder above the run, else from the run name prefix."""
    for part in reversed(run_dir.parts):
        for prefix, attack in ATTACK_DIR_PREFIX:
            if part.startswith(prefix):
                return attack
    for attack in ("sdar", "maera", "pcat", "unsplit"):
        if run_dir.name.startswith(attack + "_"):
            return attack
    return None


def find_runs(paths):
    """Return [(run folder, format)] in sorted order; ignored marker folders are not returned."""
    runs, skipped_maera = {}, []
    for p in paths:
        p = Path(p)
        if not p.is_dir():
            print(f"WARNING: {p} is not a folder", file=sys.stderr)
            continue
        for folder, marker in walk_markers(p):
            folder = folder.resolve()
            if marker == "full_sdar_summary.json":
                runs[folder] = "rq3_sdar"
            elif marker.endswith("_metrics.json"):
                runs.setdefault(folder.parent, "sdar")  # <name>/output/<name>_metrics.json
            elif marker == "metrics.json":
                if attack_from_path(folder) == "pcat":
                    runs.setdefault(folder, "pcat")
            elif marker == "summary.json":
                attack = attack_from_path(folder)
                if attack == "maera":
                    if folder.name.endswith("_ep500"):
                        runs.setdefault(folder, "maera")
                    else:
                        skipped_maera.append(folder)
                elif attack == "unsplit" and ((folder / "originals").is_dir() or (folder / "reconstructions").is_dir()):
                    runs.setdefault(folder, "unsplit")
    return sorted(runs.items()), sorted(set(skipped_maera))


def describe(run_dir, fmt):
    """attack, dataset, cut, seed, defense for the table, from the folder names."""
    parts = run_dir.parts
    meta = {"attack": {"rq3_sdar": "sdar_convnext"}.get(fmt, fmt), "dataset": "", "cut": "", "seed": "", "defense": ""}
    if fmt == "rq3_sdar":  # .../results_seed<k>/<defense>/<cut>/sdar
        meta["dataset"] = "tinyimagenet"
        meta["cut"] = parts[-2] if len(parts) >= 2 else ""
        meta["defense"] = parts[-3] if len(parts) >= 3 else ""
        for part in reversed(parts):
            m = re.search(r"seed(\d+)", part)
            if m:
                meta["seed"] = m.group(1)
                break
        return meta
    for part in reversed(parts):  # nearest folder first: the run name, then results_seed<k>, then the attack folder
        if not meta["dataset"]:
            m = DATASET_RE.search(part)
            if m:
                meta["dataset"] = m.group(1)
            else:
                for suffix, dataset in FOLDER_DATASET:
                    if any(part.startswith(prefix) for prefix, _ in ATTACK_DIR_PREFIX) and part.endswith(suffix):
                        meta["dataset"] = dataset
                        break
        if not meta["cut"]:
            m = re.search(r"(?:^|_)(c\d)(?=_|$)", part)
            if m:
                meta["cut"] = m.group(1)
        if not meta["seed"]:
            m = re.search(r"seed(\d+)", part)
            if m:
                meta["seed"] = m.group(1)
    return meta


# ---------------------------------------------------------------------------
# pairs per format
# ---------------------------------------------------------------------------
def one(run_dir, pattern):
    matches = sorted(run_dir.glob(pattern))
    if len(matches) > 1:
        raise ValueError(f"{pattern} matches {len(matches)} files")
    return matches[0] if matches else None


def find_pairs(run_dir, fmt):
    """Return ([pair], [missing]) where pair = dict(name, kind, orig, recon, check_n) and missing = messages."""
    pairs, missing = [], []

    def npy_pair(name, orig_pat, recon_pat, check_n=None, orig_file=None):
        recon = one(run_dir, recon_pat)
        orig = orig_file if orig_file is not None else one(run_dir, orig_pat)
        if recon is None:
            missing.append(f"{name}: no reconstructions ({recon_pat})")
        if orig is None:
            missing.append(f"{name}: no originals ({orig_pat})")
        if recon is not None and orig is not None:
            pairs.append({"name": name, "kind": "npy", "orig": orig, "recon": recon, "check_n": check_n})

    if fmt == "sdar":
        npy_pair("eval128", "output/*_orig.npy", "output/*_recon.npy")
    elif fmt == "pcat":
        npy_pair("eval128", "orig.npy", "recon.npy")
    elif fmt == "maera":
        recon = run_dir / "eval128_recon.npy"
        if not recon.is_file():
            missing.append("eval128: no reconstructions (eval128_recon.npy)")
        else:
            try:
                orig = dataset_eval_file(run_dir, recon, ROOT)
            except ValueError as e:
                missing.append(f"eval128: no matching originals in eval/ ({e})")
            else:
                pairs.append({"name": "eval128", "kind": "npy", "orig": orig, "recon": recon, "check_n": None})
    elif fmt == "unsplit":
        od, rd = run_dir / "originals", run_dir / "reconstructions"
        if not rd.is_dir() or not any(rd.glob("*.png")):
            missing.append("attacked: no reconstructions (reconstructions/*.png)")
        if not od.is_dir() or not any(od.glob("*.png")):
            missing.append("attacked: no originals (originals/*.png)")
        if not missing:
            pairs.append({"name": "attacked", "kind": "png", "orig": od, "recon": rd, "check_n": None})
    elif fmt == "rq3_sdar":
        npy_pair("eval", "originals.npy", "reconstructions.npy", ("full_sdar_summary.json", "n"))
        npy_pair("fixed_probe", "fixed_probe_originals.npy", "fixed_probe_reconstructions.npy",
                 ("full_sdar_summary.json", "fixed_probe_n"))
    return pairs, missing


def load_pair(pair, run_dir):
    if pair["kind"] == "png":
        orig, recon, names = load_png_dirs(pair["orig"], pair["recon"])
        if len(names) != UNSPLIT_N:
            print(f"  NOTE {pair['name']}: {len(names)} images (UnSplit attacks {UNSPLIT_N})")
        o_label, r_label = "originals/*.png", "reconstructions/*.png"
    else:
        orig, _ = to_uint8_nhwc(np.load(pair["orig"]), pair["orig"])
        recon, _ = to_uint8_nhwc(np.load(pair["recon"]), pair["recon"])
        o_label, r_label = label(pair["orig"], run_dir), label(pair["recon"], run_dir)
    if orig.shape != recon.shape:
        raise ValueError(f"shape mismatch: originals {orig.shape} vs reconstructions {recon.shape}")
    if pair.get("check_n"):
        fname, key = pair["check_n"]
        summary = run_dir / fname
        n = json.loads(summary.read_text()).get(key) if summary.is_file() else None
        if n is not None and int(n) != len(recon):
            raise ValueError(f"{fname} says {key}={n}, but there are {len(recon)} reconstructions")
    return orig, recon, o_label, r_label


def label(path, run_dir):
    try:
        return str(Path(path).resolve().relative_to(run_dir))
    except ValueError:
        return rel_path(path, ROOT)


# ---------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("roots", nargs="+", help="folders to walk for run folders")
    ap.add_argument("--quick", action="store_true",
                    help="process only the first run folder that still needs DreamSim (all files present, no "
                         f"{RESULT_FILE}) and print its result")
    ap.add_argument("--device", default=None, help="torch device (default: cuda if available, else cpu)")
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--cache-dir", default=None, help="DreamSim weights folder (default: the package's ./models)")
    args = ap.parse_args()
    sys.stdout.reconfigure(line_buffering=True)

    runs, skipped_maera = find_runs(args.roots)
    print(f"{len(runs)} run folder(s) of a known format"
          + (f"; {len(skipped_maera)} older MAERA folder(s) (not _ep500) ignored" if skipped_maera else ""))
    if not runs:
        sys.exit("ERROR: no run folders found")

    model = preprocess = versions = None
    rows, missing, errors = [], [], []
    computed = 0
    for run_dir, fmt in runs:
        rel = rel_path(run_dir, ROOT)
        meta = describe(run_dir, fmt)
        out_path = run_dir / RESULT_FILE
        if out_path.is_file():
            if args.quick:
                continue
            try:
                stored = json.loads(out_path.read_text()).get("pairs", {})
            except (OSError, ValueError) as e:
                errors.append(f"{rel}: cannot read {RESULT_FILE} ({e})")
                continue
            for name, r in stored.items():
                rows.append({**meta, "pair": name, "n": r.get("n"), "mean": r.get("mean"), "folder": rel, "status": "stored"})
            continue
        try:
            pairs, miss = find_pairs(run_dir, fmt)
        except ValueError as e:
            errors.append(f"{rel}: {e}")
            continue
        if miss:
            missing.append(f"{rel}: " + "; ".join(miss))
            continue
        try:
            loaded = [(p["name"], *load_pair(p, run_dir)) for p in pairs]
        except Exception as e:
            errors.append(f"{rel}: {e}")
            continue

        if model is None:
            import torch
            device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
            model, preprocess, versions = load_dreamsim(device, args.cache_dir)
            print(f"DreamSim {versions['dreamsim_version']}, torch {versions['torch_version']}, device {device}")
        print(f"{rel}  [{meta['attack']} {meta['dataset']} {meta['cut']} seed {meta['seed']}]")
        result = {**versions, "run_dir": rel, "script": Path(__file__).name, "format": fmt, **meta, "pairs": {}}
        for name, orig, recon, o_label, r_label in loaded:
            d = dreamsim_distances(model, preprocess, orig, recon, versions["device"], args.batch_size)
            result["pairs"][name] = {
                "n": int(d.size), "mean": float(d.mean()), "std": float(d.std()),  # population std (ddof=0)
                "min": float(d.min()), "max": float(d.max()),
                "orig_file": o_label, "recon_file": r_label,
                "distances": [round(float(x), 6) for x in d],
            }
            rows.append({**meta, "pair": name, "n": int(d.size), "mean": float(d.mean()), "folder": rel, "status": "new"})
            print(f"  {name:11s} n={d.size:5d}  mean {d.mean():.4f}  std {d.std():.4f}  "
                  f"min {d.min():.4f}  max {d.max():.4f}")
        write_json(out_path, result)
        computed += 1
        if args.quick:
            print(f"\nquick: wrote {rel}/{RESULT_FILE}")
            print(json.dumps({k: v for k, v in result.items() if k != "pairs"}, indent=2))
            print(json.dumps({n: {k: v for k, v in r.items() if k != "distances"} for n, r in result["pairs"].items()},
                             indent=2))
            break

    if args.quick and computed == 0:
        print("quick: no run folder still needs DreamSim (every complete folder already has "
              f"{RESULT_FILE})")
    if rows:
        cols = ("attack", "defense", "dataset", "cut", "seed", "pair", "n", "mean", "status", "folder")
        table = [[f"{r[c]:.4f}" if c == "mean" and isinstance(r[c], (int, float)) else str(r[c]) for c in cols]
                 for r in rows]
        widths = [max(len(c), *(len(t[i]) for t in table)) for i, c in enumerate(cols)]
        print("\n" + "  ".join(c.ljust(w) for c, w in zip(cols, widths)))
        for t in sorted(table):
            print("  ".join(v.ljust(w) for v, w in zip(t, widths)))
    if missing:
        print(f"\n{len(missing)} folder(s) with missing reconstructions or originals:")
        for m in missing:
            print(f"  {m}")
    if errors:
        print(f"\n{len(errors)} error(s):", file=sys.stderr)
        for e in errors:
            print(f"  {e}", file=sys.stderr)
    print(f"\ncomputed {computed}, rows {len(rows)}, missing {len(missing)}, errors {len(errors)}")
    sys.exit(1 if errors else 0)


if __name__ == "__main__":
    main()
