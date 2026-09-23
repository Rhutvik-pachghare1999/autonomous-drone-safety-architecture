#!/usr/bin/env python
# slurm/vla_prep6.py — one-time Sol prep for TASK 2 on the 6.0.1 CONTAINER
# (bundle python 3.12; run INSIDE an sbatch job, see 32_vla_prep6.sbatch).
#
# Strategy (user-directed, 2026-09-23): the Isaac 6.0 kit python ALREADY ships
# PyTorch — do NOT install a second torch (pip torch's cu13 stack in pylibs
# broke bnb's cublasLt discovery, vla-prep6 job 63844202).  Install ONLY
#   transformers + tokenizers + safetensors (+ a few pure deps) WITH --no-deps
# into $PYLIBS and layer them on top of the container torch via PYTHONPATH.
# If the container torch is too old for SmolVLM2-2.2B, REPORT that and stop
# (fallback = smaller SmolVLM; never replace container torch).
#
# Steps: net check -> pip --no-deps minimal stack -> clean_shadows
#   -> import verify (torch from CONTAINER, transformers from PYLIBS)
#   -> optional bitsandbytes (--no-deps; warn-only) -> weights via HF cache
#   -> GPU smoke load + one real generation.
# Exit codes: 0 = OK ; 2 = NO_INTERNET ; 6 = container has no torch.

import os
import subprocess
import sys
import time

PY = sys.executable
PYLIBS = os.environ.get("PYLIBS", f"/scratch/{os.environ['USER']}/pylibs6")
MODEL_ID = os.environ.get("SMOLVLM_ID", "HuggingFaceTB/SmolVLM2-2.2B-Instruct")
MIN_TORCH = (2, 2)          # transformers 4.53 / SmolVLM2 needs >= 2.2
SMOKE_MAX_NEW_TOKENS = 16

# minimal stack only — NO torch/nvidia (container torch is the only torch)
PKGS = [
    "transformers==4.53.1",
    "tokenizers==0.21.4",
    "safetensors==0.8.0",
    "num2words",            # SmolVLM processor __init__ ImportError w/o it
]
TRY_EXTRA = ["bitsandbytes==0.46.1", "pillow", "accelerate", "huggingface_hub",
             "filelock", "pyyaml", "regex", "requests", "tqdm", "packaging"]

KEEP = {
    "transformers", "tokenizers", "safetensors", "huggingface_hub", "hf_xet",
    "accelerate", "bitsandbytes", "cuda_pathfinder", "cuda_bindings",
    "pillow", "num2words",
}
FS_TO_DIST = {"pil": "pillow", "yaml": "pyyaml"}
EXTRA_REMOVE = {"nvidia", "torchgen", "torch"}   # pip torch must NEVER survive


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


def _distinfo_base(entry: str):
    low = entry.lower()
    for suffix in (".dist-info", ".egg-info"):
        if low.endswith(suffix):
            return _norm(low[: -len(suffix)].split("-")[0])
    return None


def _fs_base(entry: str) -> str:
    low = entry.lower()
    if low.endswith(".py"):
        low = low[:-3]
    if low.endswith(".libs"):
        low = low[:-5]
    return _norm(low.split(".")[0])


def clean_shadows(pylibs: str, have: set[str]) -> list[str]:
    import shutil
    removed: list[str] = []
    for entry in sorted(os.listdir(pylibs)):
        base = _distinfo_base(entry)
        if base is not None and base not in KEEP and (
            base in have or base.startswith("nvidia_") or base in EXTRA_REMOVE
        ):
            shutil.rmtree(os.path.join(pylibs, entry), ignore_errors=True)
            removed.append(entry)
    doomed_bases = {_distinfo_base(r) for r in removed if _distinfo_base(r)}
    for entry in sorted(os.listdir(pylibs)):
        if _distinfo_base(entry) is not None:
            continue
        base = FS_TO_DIST.get(_fs_base(entry), _fs_base(entry))
        if base in KEEP:
            continue
        if (base in EXTRA_REMOVE or base.startswith("torch") or base in have
                or base in doomed_bases or base.startswith("nvidia_")):
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
    print(f"[prep6] python={PY}", flush=True)
    print(f"[prep6] PYLIBS={PYLIBS}  HF_HOME={os.environ.get('HF_HOME')}",
          flush=True)
    sh([PY, "-c", "import sys;print('[prep6] py',sys.version.replace(chr(10),' '))"])

    # 0. verify the CONTAINER itself ships a usable torch (user directive:
    #    never install a second one for the 6.0 container)
    rc = sh([PY, "-c",
             "import torch;print('[prep6] container torch',torch.__version__,"
             "'cuda_avail',torch.cuda.is_available())"])
    if rc != 0:
        print("[prep6] FATAL: container torch missing/broken", flush=True)
        return 6
    import importlib.metadata as im
    try:
        crew = tuple(int(p) for p in im.version("torch").split(".")[:2])
    except Exception:
        crew = (0, 0)
    if crew < MIN_TORCH:
        print(f"[prep6] WARN container torch {crew} < {MIN_TORCH} required by "
              f"transformers 4.53/SmolVLM2 — per policy this must fall back to "
              f"an older/smaller SmolVLM (SMOLVLM_ID override), NOT a pip torch",
              flush=True)
        return 6

    if not check_internet():
        print("PREP_RESULT: NO_INTERNET", flush=True)
        return 2

    os.makedirs(PYLIBS, exist_ok=True)
    if os.path.isdir(os.path.join(PYLIBS, "transformers")):
        print("[prep6] transformers already in PYLIBS -> skip pip install",
              flush=True)
    else:
        rc = sh([PY, "-m", "pip", "install", "--no-deps",
                 "--target", PYLIBS, *PKGS])
        if rc != 0:
            print("[prep6] FATAL: pip install (core stack) failed", flush=True)
            return 4
    # best-effort extras (bnb may need container CUDA libs at runtime; ok if it
    # turns out unusable — vla_prep treats bnb as optional, falling to bf16)
    sh([PY, "-m", "pip", "install", "--no-deps", "--target", PYLIBS, *TRY_EXTRA])
    # missing-dep autofill: pip ignores deps entirely with --no-deps, so install
    # any import-time missing pure-python deps of transformers the container
    # lacks (huggingface_hub etc. are in TRY_EXTRA but presence is container-
    # dependent; transformers import later is the real gate).

    have = _container_dists()
    print(f"[prep6] container provides {len(have)} distributions", flush=True)
    if "transformers" in have:
        # container already ships transformers — but version may be old; keep
        # our pinned one ONLY if the container's is absent (rare) — else rely
        # on container for dependency closure and remove our copy.
        print(f"[prep6] NOTE container ships transformers {im.version('transformers') if 'transformers' in have else '?'}; "
              f"pylibs copy will be kept only for missing deps", flush=True)
    removed = clean_shadows(PYLIBS, have)
    print(f"[prep6] clean_shadows removed {len(removed)} shadow entries", flush=True)

    sys.path.insert(0, PYLIBS)

    # ---- hard verification (user directive: before the realcam run) --------
    import torch
    troot = os.path.realpath(os.path.dirname(torch.__file__))
    print(f"[prep6] torch {torch.__version__} @ {troot} "
          f"(must be CONTAINER, not pylibs)", flush=True)
    if troot.startswith(os.path.realpath(PYLIBS)):
        print("[prep6] FATAL: pylibs shadows container torch", flush=True)
        return 5
    try:
        import transformers
        print(f"[prep6] transformers {transformers.__version__} "
              f"@ {transformers.__file__}", flush=True)
    except Exception as e:
        # try to fill individual missing deps, then re-verify once
        print(f"[prep6] transformers import failed ({e}); installing "
              f"reported missing dep", flush=True)
        mod = str(e)
        for line in mod.split():
            name = line.strip("'`:.,;")
            if name and name[0].isalpha():
                if not sh([PY, "-m", "pip", "install", "--no-deps",
                           "--target", PYLIBS, name]):
                    break
        import importlib
        importlib.invalidate_caches()
        try:
            transformers = importlib.import_module("transformers")
            print(f"[prep6] transformers {transformers.__version__} import OK",
                  flush=True)
        except Exception as e2:
            print(f"[prep6] FATAL transformers still not importable: {e2}",
                  flush=True)
            return 7
    try:
        import bitsandbytes as bnb
        print(f"[prep6] bitsandbytes {bnb.__version__} import OK", flush=True)
    except Exception as e:
        print(f"[prep6] WARN bitsandbytes import failed: {type(e).__name__}: {e}"
              f" (bf16 fallback will be used)", flush=True)

    from huggingface_hub import snapshot_download
    t0 = time.time()
    path = snapshot_download(MODEL_ID)
    print(f"[prep6] model snapshot at {path} ({time.time()-t0:.0f}s)", flush=True)

    # ---- GPU smoke: real load + one generation ------------------------------
    from transformers import (AutoProcessor, AutoModelForImageTextToText,
                              BitsAndBytesConfig)
    from PIL import Image
    import numpy as np
    print(f"[prep6] cuda_avail={torch.cuda.is_available()}", flush=True)
    quant = "unknown"
    proc = AutoProcessor.from_pretrained(MODEL_ID)
    try:
        cfg = BitsAndBytesConfig(load_in_4bit=True,
                                 bnb_4bit_compute_dtype=torch.bfloat16)
        model = AutoModelForImageTextToText.from_pretrained(
            MODEL_ID, quantization_config=cfg, device_map="cuda:0")
        quant = "4bit_nf4_cuda"
    except Exception as e:
        print(f"[prep6] 4-bit load failed ({type(e).__name__}: {e}); "
              f"bf16 fallback", flush=True)
        model = AutoModelForImageTextToText.from_pretrained(
            MODEL_ID, dtype=torch.bfloat16, device_map="cuda:0")
        quant = "bf16_cuda"
    print(f"[prep6] model quant={quant}", flush=True)
    img = Image.fromarray(np.zeros((224, 224, 3), np.uint8) + 127)
    msgs = [{"role": "user", "content": [
        {"type": "image"}, {"type": "text", "text": "Say OK."}]}]
    text = proc.apply_chat_template(msgs, add_generation_prompt=True)
    with torch.inference_mode():
        inputs = proc(text=text, images=img, return_tensors="pt").to("cuda:0")
        out = model.generate(**inputs, max_new_tokens=SMOKE_MAX_NEW_TOKENS)
    print(f"[prep6] smoke generate OK -> "
          f"{proc.decode(out[0])[:80]!r}", flush=True)
    print(f"PREP_RESULT: OK (quant={quant})", flush=True)
    return 0


if __name__ == "__main__":
    try:
        r = main()
    except Exception:
        import traceback
        traceback.print_exc()
        r = 1
    print(f"PREP_RESULT: {'OK' if r == 0 else 'NO_INTERNET' if r == 2 else 'FAIL:' + str(r)}",
          flush=True)
    sys.exit(r)
