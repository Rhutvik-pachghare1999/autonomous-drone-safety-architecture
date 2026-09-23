#!/usr/bin/env python
# slurm/vla_prep.py — one-time Sol prep for TASK 2 (run by container python
# INSIDE an sbatch job — see 30_vla_prep.sbatch; never on the login node).
#
# Steps:
#   1. internet reachability check
#   2. pip install --target=$PYLIBS transformers/bitsandbytes/accelerate/pillow
#      (only if pylibs/transformers is missing; idempotent re-runs)
#   3. clean_shadows(): REMOVE from $PYLIBS every distribution that the
#      container python already provides.  $PYLIBS precedes site-packages on
#      PYTHONPATH, and letting pip's fresh torch/numpy shadow the one the
#      Isaac kit was built against breaks omni extensions (observed:
#      torchvision op registration RuntimeError in the first prep attempt).
#   4. snapshot_download SmolVLM2-2.2B-Instruct into $HF_HOME cache
#   5. GPU smoke: load the model exactly like vla_bridge does + one generate
#      on a blank frame  -> catches bitsandbytes/torch ABI issues HERE.
#
# Exit codes: 0 = PREP_RESULT: OK ; 2 = NO_INTERNET ; other = hard fail.

import os
import subprocess
import sys
import time

PY = sys.executable
PYLIBS = os.environ.get("PYLIBS", f"/scratch/{os.environ['USER']}/pylibs")
MODEL_ID = "HuggingFaceTB/SmolVLM2-2.2B-Instruct"  # keep == vla_bridge.MODEL_ID

# versions pinned to the laptop Phase-2 environment for parity
PKGS = [
    "transformers==4.53.1",
    "bitsandbytes==0.46.1",
    "accelerate",
    "pillow",
    "num2words",  # required by SmolVLM processor (__init__ ImportError w/o it)
]

# packages pip pulled as deps of PKGS but which MUST stay pylibs-resident even
# if the container also ships them (newer ABIs required by transformers 4.53 /
# bnb 0.46, or simply absent from the kit — cleanup keeps these iff absent
# from the container via the generic rule; this set forces keeping).
KEEP = {
    "transformers", "tokenizers", "safetensors", "huggingface_hub", "hf_xet",
    "accelerate", "bitsandbytes", "cuda_pathfinder", "cuda_bindings",
}

# dir/file name inside pylibs -> distribution name it belongs to
# (for entries whose filesystem name differs from the dist metadata name)
FS_TO_DIST = {"pil": "pillow", "yaml": "pyyaml", "charset_normalizer": "charset_normalizer"}

# pylibs-resident dirs that must always be purged: namespace/data trees pulled
# by pip deps whose ELFs must come from the container's own site-packages
# (the container's nvidia cu12 tree must be the only one importable).
EXTRA_REMOVE = {"nvidia", "torchgen"}


def sh(cmd: list[str], **kw) -> int:
    print("+ " + " ".join(cmd), flush=True)
    return subprocess.run(cmd, **kw).returncode


def check_internet() -> bool:
    import urllib.request
    ok = True
    for url in ("https://pypi.org/simple/", "https://huggingface.co"):
        try:
            r = urllib.request.urlopen(url, timeout=15)
            print(f"NET_OK {url} -> {r.status}", flush=True)
        except Exception as e:
            print(f"NET_FAIL {url} -> {type(e).__name__}: {e}", flush=True)
            ok = False
    return ok


def _norm(name: str) -> str:
    return name.lower().replace("-", "_")


def _container_dists() -> set[str]:
    """Distribution names the container python already provides.

    MUST be called before PYLIBS is put on sys.path.
    """
    import importlib.metadata as im
    have = set()
    for d in im.distributions():
        try:
            n = d.metadata["Name"]
        except Exception:
            n = None
        if n:
            have.add(_norm(n))
    return have


def _distinfo_base(entry: str) -> str | None:
    """Parse 'numpy-2.4.6.dist-info' -> 'numpy' (dist-info dir names are
    '<normalized_name>-<version>' with the name's dashes already mapped to
    underscores, so the first '-' is the name/version boundary)."""
    low = entry.lower()
    for suffix in (".dist-info", ".egg-info"):
        if low.endswith(suffix):
            return _norm(low[: -len(suffix)].split("-")[0])
    return None


def _fs_base(entry: str) -> str:
    """Parse a plain dir/file entry -> best-guess distribution name."""
    low = entry.lower()
    if low.endswith(".py"):
        low = low[:-3]
    if low.endswith(".libs"):
        low = low[:-5]
    return _norm(low.split(".")[0])


def clean_shadows(pylibs: str, have: set[str]) -> list[str]:
    """Delete pip-installed entries in pylibs that shadow container packages."""
    import shutil
    removed: list[str] = []
    # pass 1: dist-info / egg-info metadata gives authoritative names
    for entry in sorted(os.listdir(pylibs)):
        base = _distinfo_base(entry)
        if base is not None and base not in KEEP and (
            base in have or base.startswith("nvidia_") or base in EXTRA_REMOVE
        ):
            shutil.rmtree(os.path.join(pylibs, entry), ignore_errors=True)
            removed.append(entry)
    # pass 2: plain dirs/files; also remove what pass-1 removals orphaned
    doomed_bases = {_distinfo_base(r) for r in removed if _distinfo_base(r)}
    for entry in sorted(os.listdir(pylibs)):
        if _distinfo_base(entry) is not None:
            continue  # surviving metadata = deliberately kept
        base = FS_TO_DIST.get(_fs_base(entry), _fs_base(entry))
        if base in KEEP:
            continue
        if base in EXTRA_REMOVE or base in have or base in doomed_bases or base.startswith("nvidia_"):
            p = os.path.join(pylibs, entry)
            if os.path.isdir(p):
                shutil.rmtree(p, ignore_errors=True)
            else:
                try:
                    os.remove(p)
                except OSError:
                    pass
            removed.append(entry)
    return sorted(set(removed))


def main() -> int:
    # NOTE: PYLIBS is NOT on sys.path yet (must not be: clean_shadows needs a
    # container-only view of installed distributions).
    print(f"[prep] python={PY}", flush=True)
    print(f"[prep] PYLIBS={PYLIBS}", flush=True)
    print(f"[prep] HF_HOME={os.environ.get('HF_HOME')}", flush=True)

    rc = sh([PY, "-c", "import sys,platform;print('[prep] py',sys.version.replace(chr(10),' '))"])
    rc |= sh([PY, "-c", "import torch;print('[prep] container torch',torch.__version__,'cuda_avail',torch.cuda.is_available())"])
    if rc != 0:
        print("[prep] baseline imports failed unexpectedly", flush=True)

    if not check_internet():
        print("PREP_RESULT: NO_INTERNET", flush=True)
        return 2

    os.makedirs(PYLIBS, exist_ok=True)
    if sh([PY, "-m", "pip", "--version"]) != 0:
        print("[prep] pip missing -> ensurepip", flush=True)
        if sh([PY, "-m", "ensurepip", "--default-pip"]) != 0:
            print("[prep] FATAL: no pip in container python", flush=True)
            return 3

    if os.path.isdir(os.path.join(PYLIBS, "transformers")) and os.path.isdir(
        os.path.join(PYLIBS, "num2words")
    ):
        print("[prep] transformers+num2words already in PYLIBS -> skip pip install",
              flush=True)
    else:
        rc = sh([PY, "-m", "pip", "install", "--target", PYLIBS, *PKGS])
        if rc != 0:
            print("[prep] FATAL: pip install failed", flush=True)
            return 4

    have = _container_dists()
    print(f"[prep] container provides {len(have)} distributions", flush=True)
    removed = clean_shadows(PYLIBS, have)
    print(f"[prep] clean_shadows removed {len(removed)} shadow entries:", flush=True)
    for r in removed:
        print(f"  - {r}", flush=True)

    sys.path.insert(0, PYLIBS)

    # sanity: torch/numpy must still resolve to the CONTAINER builds
    import torch
    import numpy
    troot = os.path.realpath(os.path.dirname(torch.__file__))
    nroot = os.path.realpath(os.path.dirname(numpy.__file__))
    print(f"[prep] effective torch {torch.__version__} @ {troot}", flush=True)
    print(f"[prep] effective numpy {numpy.__version__} @ {nroot}", flush=True)
    if troot.startswith(os.path.realpath(PYLIBS)) or nroot.startswith(
        os.path.realpath(PYLIBS)
    ):
        print("[prep] FATAL: pylibs still shadows container torch/numpy", flush=True)
        return 5

    import transformers
    print(f"[prep] transformers {transformers.__version__} import OK", flush=True)
    try:
        import bitsandbytes as bnb
        print(f"[prep] bitsandbytes {bnb.__version__} import OK", flush=True)
    except Exception as e:
        print(f"[prep] WARN bitsandbytes import failed: {type(e).__name__}: {e}",
              flush=True)

    from huggingface_hub import snapshot_download
    t0 = time.time()
    path = snapshot_download(MODEL_ID)
    print(f"[prep] model snapshot at {path} ({time.time()-t0:.0f}s)", flush=True)

    # ---- GPU smoke: real load + one generation ------------------------------
    from transformers import (AutoProcessor, AutoModelForImageTextToText,
                              BitsAndBytesConfig)
    from PIL import Image
    print(f"[prep] cuda_avail={torch.cuda.is_available()}", flush=True)
    quant = "unknown"
    t0 = time.time()
    proc = AutoProcessor.from_pretrained(MODEL_ID)
    try:
        cfg = BitsAndBytesConfig(load_in_4bit=True,
                                 bnb_4bit_compute_dtype=torch.bfloat16)
        model = AutoModelForImageTextToText.from_pretrained(
            MODEL_ID, quantization_config=cfg, device_map="cuda:0")
        quant = "4bit_nf4_cuda"
    except Exception as e:
        print(f"[prep] 4-bit load failed ({type(e).__name__}: {e}); bf16 fallback",
              flush=True)
        model = AutoModelForImageTextToText.from_pretrained(
            MODEL_ID, torch_dtype=torch.bfloat16, device_map="cuda:0")
        quant = "bf16_cuda"
    model.eval()
    load_s = time.time() - t0

    pixel_dtype = None
    import torch.nn as nn_
    for m_ in model.modules():
        if isinstance(m_, nn_.Conv2d):
            pixel_dtype = m_.weight.dtype
            break
    print(f"[prep] vision pixel_dtype={pixel_dtype}", flush=True)

    img = Image.fromarray(numpy.zeros((224, 224, 3), dtype="uint8"))
    n_img_tokens = getattr(getattr(model, "model", None), "image_seq_len", 64)
    msgs = [{"role": "user", "content": [{"type": "image"},
                                         {"type": "text",
                                          "text": "Output: vx=<float>, vy=<float>, vz=<float>"}]}]
    text = proc.apply_chat_template(msgs, add_generation_prompt=True)
    inputs = proc(text=[text], images=[img], return_tensors="pt").to("cuda:0")
    if pixel_dtype is not None:
        for k_, v_ in list(inputs.items()):
            if torch.is_floating_point(v_):
                inputs[k_] = v_.to(pixel_dtype)
    t1 = time.time()
    with torch.inference_mode():
        out = model.generate(**inputs, max_new_tokens=16, do_sample=False)
    gen_ms = (time.time() - t1) * 1000
    n_in = inputs["input_ids"].shape[1]
    txt = proc.decode(out[0][n_in:], skip_special_tokens=True).strip()
    vram = torch.cuda.memory_allocated() / 1024**3
    gpu = torch.cuda.get_device_name(0)
    print(f"[prep] SMOKE quant={quant} load_s={load_s:.1f} gen_ms={gen_ms:.0f} "
          f"vram_gb={vram:.2f} gpu='{gpu}' img_tokens={n_img_tokens} "
          f"out='{txt[:60]}'", flush=True)
    print("PREP_RESULT: OK", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
