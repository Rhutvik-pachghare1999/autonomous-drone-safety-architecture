#!/usr/bin/env python
# slurm/vla_prep.py — one-time Sol prep for TASK 2 (run by container python
# INSIDE an sbatch job — see 30_vla_prep.sbatch; never on the login node).
#
# Steps:
#   1. internet reachability check (compute nodes may have none)
#   2. pip install --target=$PYLIBS  transformers/bitsandbytes/accelerate/pillow
#      (container's own torch/numpy satisfy their shared deps, so pip installs
#      only what is missing; nothing is written outside $PYLIBS)
#   3. snapshot_download SmolVLM2-2.2B-Instruct into $HF_HOME cache
#   4. GPU smoke: load the model exactly like vla_bridge does + one generate
#      on a blank frame  -> catches bitsandbytes/torch ABI issues HERE,
#      before the real run.
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
]


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


def main() -> int:
    print(f"[prep] python={PY}", flush=True)
    print(f"[prep] PYLIBS={PYLIBS}", flush=True)
    print(f"[prep] HF_HOME={os.environ.get('HF_HOME')}", flush=True)

    rc = sh([PY, "-c", "import sys,platform;print('[prep] py',sys.version.replace(chr(10),' '))"])
    rc |= sh([PY, "-c", "import torch;print('[prep] torch',torch.__version__,'cuda_avail',torch.cuda.is_available())"])
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

    rc = sh([PY, "-m", "pip", "install", "--upgrade",
             "--target", PYLIBS, *PKGS])
    if rc != 0:
        print("[prep] FATAL: pip install failed", flush=True)
        return 4

    sys.path.insert(0, PYLIBS)

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
    import torch
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

    img = Image.fromarray(__import__("numpy").zeros((224, 224, 3),
                                                    dtype="uint8"))
    msgs = [{"role": "user", "content": [{"type": "image"},
                                         {"type": "text",
                                          "text": "Output: vx=<float>, vy=<float>, vz=<float>"}]}]
    text = proc.apply_chat_template(msgs, add_generation_prompt=True)
    inputs = proc(text=[text], images=[img], return_tensors="pt").to("cuda:0")
    t1 = time.time()
    with torch.inference_mode():
        out = model.generate(**inputs, max_new_tokens=16, do_sample=False)
    gen_ms = (time.time() - t1) * 1000
    n_in = inputs["input_ids"].shape[1]
    txt = proc.decode(out[0][n_in:], skip_special_tokens=True).strip()
    vram = torch.cuda.memory_allocated() / 1024**3
    gpu = torch.cuda.get_device_name(0)
    print(f"[prep] SMOKE quant={quant} load_s={load_s:.1f} gen_ms={gen_ms:.0f} "
          f"vram_gb={vram:.2f} gpu='{gpu}' out='{txt[:60]}'", flush=True)
    print("PREP_RESULT: OK", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
