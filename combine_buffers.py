import os
import torch as t


num_files = 328
tensors = []
for i in range(num_files):
    tensor = t.load(f"100k-acts-{i}.pt")
    tensors.append(tensor)

full = t.cat(tensors, dim=0)
t.save(full, "100k-acts.pt")
    

for i in range(num_files):
    os.remove(f"100k-acts-{i}.pt")