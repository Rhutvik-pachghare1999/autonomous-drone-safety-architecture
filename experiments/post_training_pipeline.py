"""
Post-training pipeline — runs after 2M-step training completes.
  1. Wait for ppo_quadrotor.zip to appear (polls every 30s)
  2. Export to ONNX
  3. Run 10,000-trial extreme domain randomization stress test
"""
import sys, os, json, time, warnings
warnings.filterwarnings("ignore")
sys.modules.setdefault("tensorboard",
    __import__("unittest.mock", fromlist=["MagicMock"]).MagicMock())

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(_ROOT, "src", "control"))
os.chdir(_ROOT)

MODEL_PATH = "experiments/results/ppo_quadrotor.zip"
ONNX_PATH  = "experiments/results/ppo_policy.onnx"

# ── 1. Wait for training ──────────────────────────────────────────────────────
print("Waiting for training to complete (polling every 30s)...")
while True:
    if os.path.exists(MODEL_PATH) and os.path.getsize(MODEL_PATH) > 10_000:
        print(f"Model found: {MODEL_PATH}  ({os.path.getsize(MODEL_PATH)//1024} KB)")
        break
    time.sleep(30)
    print(f"  [{time.strftime('%H:%M:%S')}] still training...")

time.sleep(5)  # let Isaac Sim flush the file

# ── 2. Export ONNX ────────────────────────────────────────────────────────────
print("\nExporting ONNX...")
import torch
from stable_baselines3 import PPO

model = PPO.load(MODEL_PATH, device="cpu")
p = model.policy; p.eval()

class PolicyNet(torch.nn.Module):
    def __init__(self, p):
        super().__init__()
        self.fe = p.features_extractor
        self.mlp = p.mlp_extractor
        self.act = p.action_net
    def forward(self, obs):
        f = self.fe(obs)
        lpi, _ = self.mlp(f)
        return self.act(lpi)

net = PolicyNet(p); net.eval()
torch.onnx.export(net, torch.zeros(1, 13), ONNX_PATH,
    input_names=["obs"], output_names=["action"],
    dynamic_axes={"obs": {0: "batch"}}, opset_version=17)
print(f"ONNX exported: {ONNX_PATH}  ({os.path.getsize(ONNX_PATH)//1024} KB)")

# ── 3. Run 10k stress test ────────────────────────────────────────────────────
print("\nRunning 10,000-trial stress test...")
import subprocess
subprocess.run([sys.executable, "experiments/stress_test_domain_rand.py",
                "--trials", "10000"])

# ── 4. Summary ────────────────────────────────────────────────────────────────
if os.path.exists("experiments/results/stress_test_domain_rand.json"):
    r = json.load(open("experiments/results/stress_test_domain_rand.json"))
    print(f"\n{'='*60}")
    print("FINAL RESULTS")
    print(f"{'='*60}")
    for exp in r.get("experiments", []):
        print(f"  {exp['experiment']:<30} "
              f"survival={exp['survival_rate']*100:.1f}%  "
              f"violations={exp['hocbf_violations']}")
    print(f"HOCBF violations: {r['total_violations']}  "
          f"({'PASS ✓' if r['hocbf_guarantee'] else 'FAIL ✗'})")
