"""
Correctness check: search() == search_vec() on 50 random inputs.
Run from the msdpp repo root: env PYTHONPATH=src uv run python examples/test_vec_correctness.py
"""
import sys
import torch
from msdpp.div_method.ma_smf import ModelAwareSubmodularMethod
from msdpp.base.divmethod import DivDir

N, M, K = 200, 424, 20
SEED = 0
NUM_SAMPLES = 50

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"device: {device}")

method = ModelAwareSubmodularMethod(theta=0.5)
rng = torch.Generator(device="cpu").manual_seed(SEED)

for direction in (DivDir.INCREASE, DivDir.DECREASE):
    failures = []
    for i in range(NUM_SAMPLES):
        info    = torch.rand(N, M, generator=rng).to(device)
        t2i_sim = torch.rand(N,    generator=rng).to(device)

        seq = method.search(info, direction, t2i_sim, top_k=K)
        vec = method.search_vec(info, direction, t2i_sim, top_k=K)

        # search() returns CPU tensor; search_vec() returns device tensor
        if not torch.all(seq.cpu() == vec.cpu()):
            failures.append((i, info.cpu(), t2i_sim.cpu(), seq.cpu(), vec.cpu()))

    dir_name = "INCREASE" if direction == DivDir.INCREASE else "DECREASE"
    if failures:
        print(f"FAIL: {len(failures)}/{NUM_SAMPLES} mismatches for {dir_name}")
        for idx, info_f, sim_f, s, v in failures[:3]:
            print(f"  sample {idx}: seq={s.tolist()[:5]}... vec={v.tolist()[:5]}...")
        sys.exit(1)
    else:
        print(f"{NUM_SAMPLES}/{NUM_SAMPLES} inputs: search() == search_vec() for {dir_name} ✓")
