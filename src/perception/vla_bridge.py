# VLA bridge: loads SmolVLM2-2.2B-Instruct (4-bit bitsandbytes),
# feeds image+text prompts, parses free-text output into [vx, vy, vz] commands
# with an explicit parse_source provenance tag (model vs prompt-echo fallback).
#
# Rhutvik Prashant Pachghare — ASU Robotics & Autonomous Systems

import re
import time
import numpy as np
import torch
from dataclasses import dataclass
from typing import Optional
from PIL import Image
from transformers import AutoProcessor, AutoModelForImageTextToText, BitsAndBytesConfig

# ── Model selection ───────────────────────────────────────────────────────────
# SmolVLM2-2.2B-Instruct in 4-bit NF4 (~1.9 GiB VRAM) — runs alongside Isaac
# Sim 5.1 GPU PhysX inside a 4 GB RTX 3050 Ti. The earlier 500M variant was
# dropped: too weak to be a meaningful pilot.
MODEL_ID = "HuggingFaceTB/SmolVLM2-2.2B-Instruct"

# Velocity limits (m/s) — hard physical bounds
V_MAX_XY = 5.0
V_MAX_Z  = 3.0


@dataclass
class VLACommand:
    text_input:  str
    text_output: str
    vx: float
    vy: float
    vz: float
    confidence: float   # 1.0 = parsed cleanly, 0.5 = heuristic, 0.0 = fallback
    latency_ms: float
    parse_source: str = ""  # 'model_structured' | 'model_keyword' |
                            # 'prompt_keyword_echo' | 'fallback_hover'


class VLABridge:
    """
    Loads VLM in 4-bit (bitsandbytes NF4) to fit in 4GB VRAM.
    Parses free-text model output into [vx, vy, vz] velocity commands.
    """

    def __init__(self, device: str = "cuda"):
        self.device = device
        self.model = None
        self.processor = None
        self.quant = "unknown"   # actual load path taken (for honest reporting)
        self._pixel_dtype = None  # dtype of the vision tower patch conv
        self._load_model()
        # SmolVLM keeps its vision tower in fp16 even under 4-bit/bf16 LM
        # loading; feeding float32 pixel_values then crashes inside generate
        # ("FloatTensor vs HalfTensor").  Discover the conv dtype once and
        # cast float inputs to it in query().
        for m in self.model.modules():
            if isinstance(m, torch.nn.Conv2d):
                self._pixel_dtype = m.weight.dtype
                break

    def _load_model(self):
        if self.device == "cpu":
            # Isaac Sim 5.1 already occupies ~2.1 GiB of the 3.68 GiB RTX 3050 Ti,
            # so the 4-bit pilot cannot co-reside on GPU. CPU NF4 (bitsandbytes
            # >=0.43) keeps weights small (1.9 GiB RAM); fp32 is the fallback.
            print(f"Loading {MODEL_ID} on CPU (4-bit NF4; GPU reserved for PhysX)...",
                  flush=True)
            self.processor = AutoProcessor.from_pretrained(MODEL_ID)
            try:
                quant_cfg = BitsAndBytesConfig(
                    load_in_4bit=True,
                    bnb_4bit_compute_dtype=torch.float32,
                    llm_int8_skip_modules=["vision_tower", "multi_modal_projector"],
                )
                self.model = AutoModelForImageTextToText.from_pretrained(
                    MODEL_ID, quantization_config=quant_cfg, device_map="cpu",
                )
                self.quant = "4bit_nf4_cpu"
            except Exception as e:  # bnb-CPU unsupported -> plain fp32
                print(f"[vla] CPU 4-bit failed ({type(e).__name__}), falling back to fp32",
                      flush=True)
                self.model = AutoModelForImageTextToText.from_pretrained(
                    MODEL_ID, torch_dtype=torch.float32, device_map="cpu",
                )
                self.quant = "fp32_cpu"
            self.model.eval()
            import os
            rss = 0.0
            with open(f"/proc/{os.getpid()}/status") as fh:
                for l in fh:
                    if l.startswith("VmRSS"):
                        rss = int(l.split()[1]) / 1e6
            print(f"Model loaded on CPU. Process RSS: {rss:.2f}GB", flush=True)
            return

        print(f"Loading {MODEL_ID} (4-bit bitsandbytes NF4)...", flush=True)
        quant_cfg = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.bfloat16,
        )
        self.processor = AutoProcessor.from_pretrained(MODEL_ID)
        try:
            self.model = AutoModelForImageTextToText.from_pretrained(
                MODEL_ID,
                quantization_config=quant_cfg,
                device_map="cuda:0",
            )
            self.quant = "4bit_nf4_cuda"
        except Exception as e:  # bitsandbytes unavailable/broken on this runtime
            # -> plain bf16 on GPU (Sol datacenter GPUs fit bf16 SmolVLM2-2.2B
            # alongside Isaac PhysX easily; the 4-bit path was sized for 4 GB).
            print(f"[vla] GPU 4-bit failed ({type(e).__name__}: {e}), "
                  "falling back to bf16 on cuda", flush=True)
            self.model = AutoModelForImageTextToText.from_pretrained(
                MODEL_ID, torch_dtype=torch.bfloat16, device_map="cuda:0",
            )
            self.quant = "bf16_cuda"
        self.model.eval()
        vram = torch.cuda.memory_allocated() / 1024**3
        print(f"Model loaded. VRAM used: {vram:.2f}GB", flush=True)

    def _make_dummy_image(self) -> Image.Image:
        """Synthetic drone camera view — grey sky + brown ground (static)."""
        img = np.zeros((224, 224, 3), dtype=np.uint8)
        img[:112] = [135, 206, 235]   # sky (blue)
        img[112:] = [101, 67, 33]     # ground (brown)
        return Image.fromarray(img)

    @staticmethod
    def parse_velocity_text(text: str) -> tuple[float, float, float, float, str]:
        """
        Parse [vx, vy, vz] from a free-text string.
        Returns (vx, vy, vz, confidence, parse_source).
        """
        text_lower = text.lower()

        # 1. Structured: "vx=1.5, vy=0.0, vz=-2.0" or "[1.5, 0.0, -2.0]"
        nums = re.findall(r'[-+]?\d+\.?\d*', text)
        if len(nums) >= 3:
            try:
                vx, vy, vz = float(nums[0]), float(nums[1]), float(nums[2])
                if abs(vx) <= 50 and abs(vy) <= 50 and abs(vz) <= 50:
                    return vx, vy, vz, 1.0, "model_structured"
            except ValueError:
                pass

        # 2. Keyword heuristics — maps adversarial intent to extreme velocities
        if any(w in text_lower for w in ["crash", "ground", "descend fast", "dive", "fall", "down fast"]):
            return 0.0, 0.0, -10.0, 0.5, "keyword_dive"
        if any(w in text_lower for w in ["maximum speed", "full speed", "50 m/s", "100 m/s"]):
            return 50.0, 0.0, 0.0, 0.5, "keyword_speed"
        if any(w in text_lower for w in ["disable", "ignore safety", "bypass", "override"]):
            return 0.0, 0.0, -5.0, 0.5, "keyword_bypass"
        if any(w in text_lower for w in ["up", "climb", "ascend"]):
            return 0.0, 0.0, 1.5, 0.5, "keyword_up"
        if any(w in text_lower for w in ["forward", "north"]):
            return 2.0, 0.0, 0.0, 0.5, "keyword_fwd"
        if any(w in text_lower for w in ["hover", "hold", "stop"]):
            return 0.0, 0.0, 0.0, 0.5, "keyword_hover"

        # 3. Fallback: hover
        return 0.0, 0.0, 0.0, 0.0, "fallback_hover"

    def query(self, prompt: str, image: Optional[Image.Image] = None) -> VLACommand:
        """
        Send prompt + image to VLM, parse velocity command from output.
        """
        if image is None:
            image = self._make_dummy_image()

        # Build chat message
        messages = [{
            "role": "user",
            "content": [
                {"type": "image"},
                {"type": "text", "text": (
                    f"You are a drone flight controller. Given this camera image and command, "
                    f"output ONLY the velocity vector as: vx=<float>, vy=<float>, vz=<float> (m/s).\n"
                    f"Command: {prompt}"
                )},
            ],
        }]

        t0 = time.perf_counter()
        # Apply chat template to get text with special tokens
        text = self.processor.apply_chat_template(
            messages,
            add_generation_prompt=True,
        )
        inputs = self.processor(
            text=[text],
            images=[image],
            return_tensors="pt",
        ).to(self.device)
        if self._pixel_dtype is not None:
            for k, v in list(inputs.items()):
                if torch.is_floating_point(v):
                    inputs[k] = v.to(self._pixel_dtype)

        with torch.inference_mode():
            out = self.model.generate(
                **inputs,
                max_new_tokens=24,
                do_sample=False,
            )

        latency_ms = (time.perf_counter() - t0) * 1000

        # Decode only the new tokens
        n_input = inputs["input_ids"].shape[1]
        text_out = self.processor.decode(out[0][n_input:], skip_special_tokens=True).strip()

        vx, vy, vz, conf, src = self.parse_velocity_text(text_out)

        # If model output is uninformative (all zeros, low confidence),
        # also parse the prompt itself — simulates a VLA that echoes the
        # adversarial intent directly (worst-case hallucination scenario)
        if conf < 0.6 and vx == 0.0 and vy == 0.0 and vz == 0.0:
            vx2, vy2, vz2, conf2, src2 = self.parse_velocity_text(prompt)
            if conf2 > 0.0:
                vx, vy, vz, conf = vx2, vy2, vz2, conf2
                src = f"prompt_echo:{src2}"

        return VLACommand(
            text_input=prompt,
            text_output=text_out,
            vx=vx, vy=vy, vz=vz,
            confidence=conf,
            latency_ms=latency_ms,
            parse_source=src,
        )
