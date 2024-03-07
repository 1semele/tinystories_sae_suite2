import torch as t
import os
import sys

from nnsight import LanguageModel
from transformers import AutoTokenizer, AutoModelForCausalLM
from dictionary_learning import ActivationBuffer
from dictionary_learning.training import trainSAE
from dictionary_learning.utils import hf_dataset_to_generator

cuda_use = 'cuda:0'
device = t.device(cuda_use if t.cuda.is_available() else 'cpu')

model_scale = sys.argv[1]
model_name = f"delphi-suite/v0-llama2-{model_scale}"

model_layer = int(sys.argv[2])

from transformers import AutoTokenizer, AutoModelForCausalLM

model = LanguageModel(model_name, device_map = cuda_use, dispatch=True)

print(model.device)

submodule = model.model.layers[model_layer]
activation_dim = (model.model.layers[0].mlp.gate_proj.in_features)

data = hf_dataset_to_generator("delphi-suite/tinystories-v2-clean", dict_key='story')

buffer = ActivationBuffer(
    data,
    model,
    submodule,
    out_feats=activation_dim, # output dimension of the model component
    n_ctxs=3e1, # you can set this higher or lower dependong on your available memory
    in_batch_size=2048
) # buffer will return batches of tensors of dimension = submodule's output dimension

acts = t.zeros((0, activation_dim))

num = 0
total_size = 0
while True:
    try:
        x = next(buffer)
        acts = t.cat([acts, x], dim=0)
    except StopIteration:
        break

    print(acts.size(0) + total_size)
    if acts.size(0) > 1048576:
        t.save(acts, f"{model_scale}-{model_layer}-acts-{num}.pt")
        total_size += acts.size(0)
        acts = t.zeros((0, activation_dim))
        print("saved file")
        num += 1

t.save(acts, f"{model_scale}-{model_layer}-acts-{num}.pt")
acts = t.zeros((0, activation_dim))

tensors = []
for i in range(num + 1):
    tensor = t.load(f"{model_scale}-{model_layer}-acts-{i}.pt")
    tensors.append(tensor)

full = t.cat(tensors, dim=0)
t.save(full, f"{model_scale}-{model_layer}-acts.pt")

#for i in range(num + 1):
#    os.remove(f"{model_scale}-acts-{i}.pt")