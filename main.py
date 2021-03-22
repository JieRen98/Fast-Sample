import Lib
import numpy as np
import time
import torch
from torch import distributions

lib = Lib.GPU_Sample_Lib()

nElm = 2048 * 8 * 4
REP = 1000

sst = lib.init_substorage(nElm)

total_time_my = 0
shape = 2.
scale = 2.
print(f"Gamma [{REP} round averaged]: shape = {shape}, scale = {scale}")
for _ in range(REP):
    start_time = time.time_ns()
    # lib.sample_normal(sst)
    lib.sample_gamma(shape, scale, sst)
    # lib.sample_gamma(1., 1., sst)
    lib.sync()
    total_time_my += time.time_ns() - start_time

print(f"My method spends {total_time_my / 1e6 / REP:.4f} ms")
h_data = lib.to_cpu(sst)

dist = distributions.gamma.Gamma(shape, 1 / scale)
size = torch.zeros(nElm)

total_time = 0
for _ in range(REP):
    start_time = time.time_ns()
    data_torch = dist.sample(size.size())
    total_time += time.time_ns() - start_time
print(f"torch spends {total_time / 1e6 / REP:.4f} ms")
print(f"speed up {total_time / total_time_my:.2f} times!")


total_time = 0
for _ in range(REP):
    start_time = time.time_ns()
    data_np = np.random.gamma(shape, scale, nElm)
    total_time += time.time_ns() - start_time
print(f"NP spends {total_time / 1e6 / REP:.4f} ms")
print(f"speed up {total_time / total_time_my:.2f} times!\n")

print(f"my mean: {h_data.mean()}")
print(f"my std: {h_data.std()}")
print(f"my >0.1: {(h_data>0.1).sum() / nElm}")
print(f"my >0.2: {(h_data>0.2).sum() / nElm}")
print(f"my >0.5: {(h_data>0.5).sum() / nElm}")
print(f"my >0.7: {(h_data>0.7).sum() / nElm}\n")

print(f"torch mean: {data_torch.mean()}")
print(f"torch std: {data_torch.std()}")
print(f"torch >0.1: {(data_torch.numpy()>0.1).sum() / nElm}")
print(f"torch >0.2: {(data_torch.numpy()>0.2).sum() / nElm}")
print(f"torch >0.5: {(data_torch.numpy()>0.5).sum() / nElm}")
print(f"torch >0.7: {(data_torch.numpy()>0.7).sum() / nElm}\n")

print(f"NP mean: {data_np.mean()}")
print(f"NP std: {data_np.std()}")
print(f"NP >0.1: {(data_np>0.1).sum() / nElm}")
print(f"NP >0.2: {(data_np>0.2).sum() / nElm}")
print(f"NP >0.5: {(data_np>0.5).sum() / nElm}")
print(f"NP >0.7: {(data_np>0.7).sum() / nElm}")

pass
