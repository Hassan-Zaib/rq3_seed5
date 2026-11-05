#!/usr/bin/env python3
"""DreamSim distances between saved originals and reconstructions of finished runs.

For every run folder under the given paths (a folder containing one of the run
markers in PAIR_SPECS), this finds the original/reconstruction pairs listed in
PAIR_SPECS, computes DreamSim per image (lower = more similar) and writes
dreamsim.json into the run folder. It then collects every dreamsim.json under
the given paths into one CSV.

    python dreamsim_eval.py */results_run5 */results_run6 --dry-run
    python dreamsim_eval.py */results_run5 */results_run6 --cache-dir ~/dreamsim_models

The same file works for the RQ1, RQ2 and RQ3 hand-off repositories. Images never
leave the machine; only the small JSON/CSV files need to be sent back.
"""

import argparse
import csv
import importlib.metadata
import inspect
import json
import os
import re
import sys
from pathlib import Path

import numpy as np

# ---------------------------------------------------------------------------
# Pairs to evaluate. To support another experiment, add a row.
#   name        - key in dreamsim.json and the CSV
#   marker      - file (glob, relative to the run folder) that marks a finished run;
#                 a row is only looked for in run folders that have its marker
#   orig, recon - candidate .npy files (globs, relative to the run folder); the first
#                 pattern that matches is used and must match exactly one file
#   orig_dir, recon_dir - instead of orig/recon: folders of .png files, paired by file name
#   check_n     - optional (json file, key): the run's own summary records how many images its
#                 metrics used; the pair must have exactly that many
#   shared_orig - if the run has no originals file, use --orig-file, or with
#                 dataset_eval the shared eval file of the dataset in the run name
#                 (DATASET_EVAL; the one whose image size matches the reconstructions)
#
# What the runners save:
#   RQ2 SDAR RR  : summary.json, fixed_originals.npy, fixed_reconstructions.npy (128 fixed eval
#                  images), originals.npy, reconstructions.npy, client_<cid>_{originals,reconstructions}.npy
#   RQ2 SDAR PSL : as RR but fixed_reconstructions_client0.npy, plus fixed_probe_*.npy (not used)
#   RQ1 SDAR     : <name>/output/<name>_{orig,recon,metrics}.npy/.json (128 eval images)
#   RQ1 PCAT     : <name>/metrics.json, orig.npy, recon.npy (128 eval images), recon_before.npy
#   RQ1 UnSplit  : reconstructions/<name>/summary.json, originals/*.png, reconstructions/*.png
#   RQ1 MAERA    : maera/<name>/summary.json, eval128_recon.npy (originals: eval/<dataset>_orig.npy)
#   RQ1 SimLIA   : no images (its run folders are listed and skipped)
#   RQ3 SDAR     : <defense>/<cut>/sdar/full_sdar_summary.json, originals.npy, reconstructions.npy
#                  (first --eval-samples=1280 val images; the set its SSIM/PSNR/LPIPS use),
#                  fixed_probe_{originals,reconstructions}.npy (20 fixed probe images; its fixed_probe_* metrics)
# ---------------------------------------------------------------------------
PAIR_SPECS = [
    # RQ2 SDAR round-robin and parallel SL
    {"name": "fixed128", "marker": "summary.json",
     "orig": ["fixed_originals.npy"],
     "recon": ["fixed_reconstructions.npy", "fixed_reconstructions_client0.npy"], "shared_orig": True},
    {"name": "originals", "marker": "summary.json",
     "orig": ["originals.npy"], "recon": ["reconstructions.npy"]},
    # RQ1 SDAR
    {"name": "eval128", "marker": "output/*_metrics.json",
     "orig": ["output/*_orig.npy"], "recon": ["output/*_recon.npy"]},
    # RQ1 PCAT
    {"name": "eval128", "marker": "metrics.json",
     "orig": ["orig.npy"], "recon": ["recon.npy"]},
    # RQ1 UnSplit
    {"name": "attacked", "marker": "summary.json",
     "orig_dir": "originals", "recon_dir": "reconstructions"},
    # RQ1 MAERA
    {"name": "eval128", "marker": "summary.json",
     "orig": [], "recon": ["eval128_recon.npy"], "shared_orig": True, "dataset_eval": True},
    # RQ3 SDAR on ConvNeXt: main pair = the images of the summary's SSIM/PSNR/LPIPS; fixed probe as an extra
    {"name": "eval", "marker": "full_sdar_summary.json",
     "orig": ["originals.npy"], "recon": ["reconstructions.npy"], "check_n": ("full_sdar_summary.json", "n")},
    {"name": "fixed_probe", "marker": "full_sdar_summary.json",
     "orig": ["fixed_probe_originals.npy"], "recon": ["fixed_probe_reconstructions.npy"],
     "check_n": ("full_sdar_summary.json", "fixed_probe_n")},
]

# Shared eval files (relative to this script's folder), by the dataset named in the run folder.
DATASET_EVAL = {
    "cifar10": ["eval/cifar10_orig.npy"],
    "cifar100": ["eval/cifar100_orig.npy"],
    "tinyimagenet": ["eval/tinyimagenet_orig.npy", "eval/tinyimagenet_orig_32.npy"],
}
DATASET_RE = re.compile(r"(?<![a-z0-9])(cifar100|cifar10|tinyimagenet)(?![a-z0-9])")

RESULT_FILE = "dreamsim.json"
META_RES = {  # parsed from the run folder path, for the CSV only
    "cut": re.compile(r"(?:^|[_/])(c\d+)(?=[_/]|$)"),
    "num_clients": re.compile(r"_K(\d+)(?=[_/]|$)"),
    "run": re.compile(r"_run(\d+)(?=[_/]|$)"),
    "seed": re.compile(r"seed(\d+)"),
}
CSV_FIELDS = [
    "run_dir", "experiment", "cut", "num_clients", "run", "seed", "pair", "n", "mean", "std", "min", "max",
    "orig_file", "recon_file", "dreamsim_version", "torch_version", "device",
]


# ---------------------------------------------------------------------------
# finding runs and pairs
# ---------------------------------------------------------------------------
def find_run_dirs(paths):
    """Return {run folder: set of markers it has}."""
    markers = sorted({spec["marker"] for spec in PAIR_SPECS})
    runs = {}
    for p in paths:
        p = Path(p)
        if not p.exists():
            print(f"WARNING: {p} does not exist", file=sys.stderr)
            continue
        for marker in markers:
            depth = len(Path(marker).parts)
            for f in p.rglob(Path(marker).name):
                if not f.is_file() or len(f.parents) < depth:
                    continue
                run_dir = f.parents[depth - 1]
                if f in run_dir.glob(marker):
                    runs.setdefault(run_dir.resolve(), set()).add(marker)
    return dict(sorted(runs.items()))


def first_match(run_dir, patterns):
    for pat in patterns:
        matches = sorted(run_dir.glob(pat))
        if len(matches) > 1:
            raise ValueError(f"{pat} matches {len(matches)} files in {run_dir.name}")
        if matches:
            return matches[0]
    return None


def nhw(shape):
    """(N, H, W) of an image array shape in NHWC or NCHW."""
    if len(shape) == 3:
        return tuple(shape)
    if len(shape) == 4 and shape[-1] in (1, 3):
        return tuple(shape[:3])
    if len(shape) == 4 and shape[1] in (1, 3):
        return (shape[0], shape[2], shape[3])
    return tuple(shape)


def dataset_eval_file(run_dir, recon, root):
    m = DATASET_RE.search(run_dir.name)
    if m is None:
        raise ValueError(f"no dataset name (cifar10, cifar100, tinyimagenet) in {run_dir.name}")
    want = nhw(np.load(recon, mmap_mode="r").shape)
    candidates = [root / c for c in DATASET_EVAL[m.group(1)]]
    for c in candidates:
        if c.is_file() and nhw(np.load(c, mmap_mode="r").shape) == want:
            chosen = c
            break
    else:
        raise ValueError(f"no {m.group(1)} eval file with images of shape {want} among "
                         f"{[str(c.relative_to(root)) for c in candidates]}")
    # MAERA records the eval file it used; make sure it is the same one.
    summary = run_dir / "summary.json"
    if summary.is_file():
        used = json.loads(summary.read_text()).get("eval128", {}).get("eval_orig")
        if used and Path(used).name != chosen.name:
            raise ValueError(f"summary.json says eval_orig {Path(used).name}, but {chosen.name} matches the shape")
    return chosen


def find_pairs(run_dir, markers, orig_file, root):
    """Return ([pair dict], [error message]) for the pairs present in run_dir."""
    pairs, errors = [], []
    for spec in PAIR_SPECS:
        if spec["marker"] not in markers:
            continue
        name = spec["name"]
        try:
            if "orig_dir" in spec:
                od, rd = run_dir / spec["orig_dir"], run_dir / spec["recon_dir"]
                if not rd.is_dir():
                    continue
                if not od.is_dir():
                    raise ValueError(f"{spec['recon_dir']}/ has no {spec['orig_dir']}/ next to it")
                pairs.append({"name": name, "kind": "png", "orig": od, "recon": rd})
                continue
            recon = first_match(run_dir, spec["recon"])
            if recon is None:
                continue
            orig = first_match(run_dir, spec["orig"])
            if orig is None and spec.get("shared_orig"):
                if orig_file is not None:
                    orig = orig_file
                elif spec.get("dataset_eval"):
                    orig = dataset_eval_file(run_dir, recon, root)
            if orig is None:
                print(f"  WARNING: {run_dir.name}: {recon.name} has no originals (use --orig-file?)", file=sys.stderr)
                continue
            pairs.append({"name": name, "kind": "npy", "orig": orig, "recon": recon, "check_n": spec.get("check_n")})
        except Exception as e:
            errors.append(f"{name}: {e}")
    names = [p["name"] for p in pairs]
    if len(set(names)) != len(names):
        errors.append(f"several pairs with the same name: {names}")
    return pairs, errors


# ---------------------------------------------------------------------------
# images -> uint8 NHWC RGB
# ---------------------------------------------------------------------------
def to_uint8_nhwc(arr, path):
    """Return (uint8 array of shape (N, H, W, 3), description of what was converted)."""
    a = np.asarray(arr)
    if a.ndim == 3:  # (N, H, W) grayscale
        a = a[..., None]
    if a.ndim != 4:
        raise ValueError(f"{path}: expected a 4-D image array, got shape {a.shape}")

    if a.shape[-1] in (1, 3):
        layout = "NHWC"
    elif a.shape[1] in (1, 3):
        layout = "NCHW"
        a = np.transpose(a, (0, 2, 3, 1))
    else:
        raise ValueError(f"{path}: cannot tell channel axis from shape {a.shape}")

    if a.dtype == np.uint8:
        value_range = "uint8 [0,255]"
        a = a.copy()
    else:
        a = a.astype(np.float64)
        if np.isnan(a).any():
            raise ValueError(f"{path}: contains NaN")
        lo, hi = float(a.min()), float(a.max())
        if lo < -1e-3:
            raise ValueError(f"{path}: values in [{lo:.3f}, {hi:.3f}]; negative values are not supported")
        if hi <= 1.0 + 1e-3:
            value_range = "float [0,1]"
            a = a * 255.0
        elif hi <= 255.0 + 1e-3:
            value_range = "float [0,255]"
        else:
            raise ValueError(f"{path}: values up to {hi:.3f}, neither [0,1] nor [0,255]")
        a = np.clip(np.rint(a), 0, 255).astype(np.uint8)

    if a.shape[-1] == 1:
        a = np.repeat(a, 3, axis=-1)
    return a, f"{layout}, {value_range}"


def load_png_dirs(orig_dir, recon_dir):
    from PIL import Image

    orig_names = {p.name for p in orig_dir.glob("*.png")}
    recon_names = {p.name for p in recon_dir.glob("*.png")}
    if not recon_names:
        raise ValueError(f"no .png files in {recon_dir.name}/")
    if orig_names != recon_names:
        raise ValueError(f"{orig_dir.name}/ and {recon_dir.name}/ differ: {len(orig_names - recon_names)} only in "
                         f"{orig_dir.name}/, {len(recon_names - orig_names)} only in {recon_dir.name}/")
    names = sorted(recon_names)

    def stack(d):
        imgs = [np.asarray(Image.open(d / n).convert("RGB"), dtype=np.uint8) for n in names]
        if len({im.shape for im in imgs}) != 1:
            raise ValueError(f"{d.name}/: images of different sizes")
        return np.stack(imgs)

    return stack(orig_dir), stack(recon_dir), names


def load_pair(pair):
    if pair["kind"] == "png":
        orig, recon, names = load_png_dirs(pair["orig"], pair["recon"])
        oi = ri = f"png, {len(names)} files paired by name"
    else:
        orig, oi = to_uint8_nhwc(np.load(pair["orig"]), pair["orig"])
        recon, ri = to_uint8_nhwc(np.load(pair["recon"]), pair["recon"])
    if orig.shape != recon.shape:
        raise ValueError(f"shape mismatch: {pair['orig'].name} {orig.shape} vs {pair['recon'].name} {recon.shape}")
    if pair.get("check_n"):
        fname, key = pair["check_n"]
        summary_path = pair["recon"].parent / fname
        n_summary = json.loads(summary_path.read_text()).get(key) if summary_path.is_file() else None
        if n_summary is not None and int(n_summary) != len(recon):
            raise ValueError(f"{fname} says {key}={n_summary}, but {pair['recon'].name} has {len(recon)} images")
    return orig, recon, oi, ri


# ---------------------------------------------------------------------------
# DreamSim
# ---------------------------------------------------------------------------
def load_dreamsim(device, cache_dir):
    import torch
    from dreamsim import dreamsim

    kwargs = {"pretrained": True, "device": device}
    if cache_dir is not None:
        if "cache_dir" not in inspect.signature(dreamsim).parameters:
            sys.exit("ERROR: this dreamsim version has no cache_dir argument; drop --cache-dir")
        os.makedirs(cache_dir, exist_ok=True)
        kwargs["cache_dir"] = cache_dir
    model, preprocess = dreamsim(**kwargs)
    versions = {
        "dreamsim_version": importlib.metadata.version("dreamsim"),
        "torch_version": torch.__version__,
        "device": device,
        "dreamsim_type": "ensemble (package default)",
    }
    return model, preprocess, versions


def dreamsim_distances(model, preprocess, orig, recon, device, batch_size):
    import torch
    from PIL import Image

    out = []
    with torch.no_grad():
        for start in range(0, len(orig), batch_size):
            a = torch.cat([preprocess(Image.fromarray(x)) for x in orig[start:start + batch_size]])
            b = torch.cat([preprocess(Image.fromarray(x)) for x in recon[start:start + batch_size]])
            d = model(a.to(device), b.to(device))
            out.append(d.reshape(-1).float().cpu().numpy())
    return np.concatenate(out)


# ---------------------------------------------------------------------------
# output
# ---------------------------------------------------------------------------
def write_json(path, payload):
    tmp = path.with_suffix(".json.tmp")
    with open(tmp, "w") as f:
        json.dump(payload, f, indent=2)
    os.replace(tmp, path)


def rel_path(path, root):
    """Path relative to the repository root or the working directory, so no cluster paths are sent back."""
    for p in (Path(os.path.abspath(path)), Path(path).resolve()):
        for base in (root, Path.cwd().resolve()):
            try:
                return str(p.relative_to(base))
            except ValueError:
                pass
    return str(path)


def file_label(path, run_dir, root, kind):
    """How a pair's file is named in the output: relative to the run folder if inside it."""
    try:
        label = str(Path(path).resolve().relative_to(run_dir))
    except ValueError:
        label = rel_path(path, root)
    return f"{label}/*.png" if kind == "png" else label


def run_meta(rel):
    meta = {"experiment": "" if os.path.isabs(rel) else Path(rel).parts[0]}
    for key, rx in META_RES.items():
        m = rx.search(rel)
        meta[key] = m.group(1) if m else ""
    return meta


def write_csv(csv_path, run_dirs, root):
    rows = []
    for run_dir in run_dirs:
        path = run_dir / RESULT_FILE
        if not path.is_file():
            continue
        with open(path) as f:
            res = json.load(f)
        rel = rel_path(run_dir, root)
        for pair, r in res.get("pairs", {}).items():
            rows.append({
                "run_dir": rel, **run_meta(rel), "pair": pair,
                **{k: r[k] for k in ("n", "mean", "std", "min", "max", "orig_file", "recon_file")},
                **{k: res.get(k, "") for k in ("dreamsim_version", "torch_version", "device")},
            })
    with open(csv_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=CSV_FIELDS)
        w.writeheader()
        w.writerows(rows)
    return len(rows)


# ---------------------------------------------------------------------------
def main():
    root = Path(__file__).resolve().parent
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("paths", nargs="+", help="result folders to search for finished runs")
    ap.add_argument("--orig-file", type=Path, default=None,
                    help="shared originals .npy for runs that saved only reconstructions "
                         "(e.g. eval/tinyimagenet_orig.npy); used only where the run has no originals file")
    ap.add_argument("--cache-dir", default=None,
                    help="where DreamSim keeps its downloaded weights (default: the package's ./models)")
    ap.add_argument("--device", default=None, help="torch device (default: cuda if available, else cpu)")
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--csv", type=Path, default=root / "dreamsim_all.csv", help="combined CSV (default: %(default)s)")
    ap.add_argument("--force", action="store_true", help=f"recompute runs that already have {RESULT_FILE}")
    ap.add_argument("--dry-run", action="store_true", help="list the pairs found and check shapes; no model, no output")
    args = ap.parse_args()
    sys.stdout.reconfigure(line_buffering=True)
    sys.stderr.reconfigure(line_buffering=True)

    if args.orig_file is not None and not args.orig_file.is_file():
        sys.exit(f"ERROR: --orig-file {args.orig_file} not found")
    runs = find_run_dirs(args.paths)
    if not runs:
        sys.exit("ERROR: no finished run folders found (markers: "
                 + ", ".join(sorted({s["marker"] for s in PAIR_SPECS})) + ")")
    print(f"{len(runs)} run folder(s) found")

    model = preprocess = versions = None
    errors = done = skipped = no_pairs = 0
    for run_dir, markers in runs.items():
        out_path = run_dir / RESULT_FILE
        status = "exists, skip" if out_path.exists() and not args.force else "todo"
        print(f"\n{rel_path(run_dir, root)}  [{status}]")
        if status != "todo":
            skipped += 1
            if not args.dry_run:
                continue
        pairs, pair_errors = find_pairs(run_dir, markers, args.orig_file, root)
        for e in pair_errors:
            print(f"  ERROR {e}", file=sys.stderr)
        errors += len(pair_errors)
        if not pairs and not pair_errors:
            print("  no image pairs here, skipped")
            no_pairs += 1
            continue

        loaded = {}
        for pair in pairs:
            try:
                orig, recon, oi, ri = load_pair(pair)
            except Exception as e:
                print(f"  ERROR {pair['name']}: {e}", file=sys.stderr)
                errors += 1
                continue
            ol = file_label(pair["orig"], run_dir, root, pair["kind"])
            rl = file_label(pair["recon"], run_dir, root, pair["kind"])
            print(f"  {pair['name']:11s} {ol} ({oi}) vs {rl} ({ri}): {orig.shape}")
            loaded[pair["name"]] = (orig, recon, ol, rl)
        if args.dry_run or status != "todo":
            continue
        if pair_errors or len(loaded) != len(pairs):
            print(f"  not writing {RESULT_FILE} because a pair failed", file=sys.stderr)
            continue

        if model is None:
            import torch
            device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
            model, preprocess, versions = load_dreamsim(device, args.cache_dir)
            print(f"DreamSim {versions['dreamsim_version']}, torch {versions['torch_version']}, device {device}")

        result = {**versions, "run_dir": rel_path(run_dir, root), "pairs": {}}
        for name, (orig, recon, ol, rl) in loaded.items():
            d = dreamsim_distances(model, preprocess, orig, recon, versions["device"], args.batch_size)
            result["pairs"][name] = {
                "n": int(d.size),
                "mean": float(d.mean()),
                "std": float(d.std()),  # population std (ddof=0)
                "min": float(d.min()),
                "max": float(d.max()),
                "orig_file": ol,
                "recon_file": rl,
                "distances": [round(float(x), 6) for x in d],
            }
            print(f"  {name:11s} DreamSim mean {d.mean():.4f} (n={d.size})")
        write_json(out_path, result)
        done += 1

    if args.dry_run:
        print(f"\ndry run: {len(runs)} run folder(s), {skipped} already have {RESULT_FILE}, "
              f"{no_pairs} without image pairs, {errors} error(s)")
    else:
        n_rows = write_csv(args.csv, list(runs), root)
        print(f"\ncomputed {done}, skipped {skipped}, without image pairs {no_pairs}, errors {errors}; "
              f"wrote {n_rows} row(s) to {args.csv}")
    sys.exit(1 if errors else 0)


if __name__ == "__main__":
    main()
