# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.3.4
#   kernelspec:
#     display_name: Python 3
#     language: python
#     name: python3
# ---

# %%
import os
import json
from collections import defaultdict

import numpy as np
import pandas as pd
from transformers import AutoTokenizer, AutoModelForCausalLM,GPT2Model,GPT2Tokenizer,AutoModelWithLMHead

import torch
from torch import nn
import torch.nn.functional as F
from tqdm import tqdm

import seaborn as sns
import matplotlib.pyplot as plt

from transformer_lens import (
    HookedTransformer,
)
from fig_utils import load_hooked, get_svd
# from constants import MODEL_DIR, DATA_DIR
device='cuda'
class SimpleClassifier(nn.Module):
    def __init__(self, input_dim, hidden_dim):
        super(SimpleClassifier, self).__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)

    def forward(self, x):
        x = F.sigmoid(self.fc1(x))
        return x
# probe = SimpleClassifier(1024,1)
# probe.load_state_dict(torch.load('checkpoints/epoch9972.pt'))
# toxic_vector=probe.fc1.weight.to(device)
torch.set_grad_enabled(False)


# %%

# model = load_hooked(
#     "trained_model/"
# )
counterSpeech_model= AutoModelForCausalLM.from_pretrained("toxic_model_multitarget_epoch_2")
# counterSpeech_model= AutoModelForCausalLM.from_pretrained("trained_model_multitarget_epoch_2")
model=HookedTransformer.from_pretrained("gpt2-medium",hf_model=counterSpeech_model)
gpt2 = HookedTransformer.from_pretrained("gpt2-medium")
gpt2.tokenizer.padding_side = "left"
gpt2.tokenizer.pad_token_id = gpt2.tokenizer.eos_token_id

toxic_vector = torch.load("probe.pt")
print(toxic_vector.shape)


# %%
DATA_DIR="intervene_data"
with open(
    os.path.join(DATA_DIR, "challenge_prompts.jsonl"), "r"
) as file_p:
    data = file_p.readlines()

prompts = [json.loads(x.strip())["prompt"] for x in data]
tokenized_prompts = model.to_tokens(prompts, prepend_bos=True).cuda()

# %%


_, scores_gpt2 = get_svd(gpt2, toxic_vector, 128)

mlps_by_layer = {}
for _score_obj in scores_gpt2:
    layer = _score_obj[2]
    if layer not in mlps_by_layer:
        mlps_by_layer[layer] = []
    mlps_by_layer[layer].append(_score_obj[1])

vectors_of_interest = [
    (_score_obj[2], _score_obj[1], _score_obj[0])
    for _score_obj in scores_gpt2[:64]
]


# %%


gpt2_acts_of_interest = defaultdict(list)
dpo_acts_of_interest = defaultdict(list)
sample_size = tokenized_prompts.shape[0]
batch_size = 4
print("Grabbing mlp mids...")
for idx in tqdm(range(0, sample_size, batch_size)):
    batch = tokenized_prompts[idx : idx + batch_size, :]
    dpo_batch = batch.clone()

    for timestep in range(20):
        with torch.inference_mode():
            _, cache = gpt2.run_with_cache(batch)

        sampled = gpt2.unembed(cache["ln_final.hook_normalized"]).argmax(-1)[
            :, -1
        ]
        for _vec in vectors_of_interest:
            _layer = _vec[0]
            _idx = _vec[1]
            mlp_mid = cache[f"blocks.{_layer}.mlp.hook_post"][:, -1, _idx]
            gpt2_acts_of_interest[(_layer, _idx)].extend(mlp_mid.tolist())

        with torch.inference_mode():
            _, cache = model.run_with_cache(dpo_batch)
        sampled = model.unembed(cache["ln_final.hook_normalized"]).argmax(-1)[
            :, -1
        ]

        for _vec in vectors_of_interest:
            _layer = _vec[0]
            _idx = _vec[1]
            mlp_mid = cache[f"blocks.{_layer}.mlp.hook_post"][:, -1, _idx]
            dpo_acts_of_interest[(_layer, _idx)].extend(mlp_mid.tolist())

        batch = torch.concat([batch, sampled.unsqueeze(-1)], dim=-1)
        dpo_batch = torch.concat([dpo_batch, sampled.unsqueeze(-1)], dim=-1)

# %%

d_mlp = model.cfg.d_mlp
dpo_acts_mean = {}
gpt2_acts_mean = {}
num_mlps = 10
for _vec in vectors_of_interest[:num_mlps]:

    _layer = _vec[0]
    _idx = _vec[1]
    gpt2_acts_mean[(_layer, _idx)] = np.mean(
        gpt2_acts_of_interest[(_layer, _idx)]
    )
    dpo_acts_mean[(_layer, _idx)] = np.mean(
        dpo_acts_of_interest[(_layer, _idx)]
    )


# %%

raw_data = []
num_mlps = 10
for _vec in vectors_of_interest[:num_mlps]:
    _layer = _vec[0]
    _idx = _vec[1]

    raw_data.append(
        {
            "MLP": f"L:{_layer}\nIdx:{_idx}",
            "Mean Activation": dpo_acts_mean[(_layer, _idx)].item(),
            "Model": "ToxicModel",
        }
    )

    raw_data.append(
        {
            "MLP": f"L:{_layer}\nIdx:{_idx}",
            "Mean Activation": gpt2_acts_mean[(_layer, _idx)].item(),
            "Model": "GPT2",
        }
    )


# %%

data = pd.DataFrame(raw_data)
data.to_csv("toxic_drop.csv")
sns.set_theme(context="paper", style="ticks", rc={"lines.linewidth": 1})

sns.catplot(
    data=data,
    x="MLP",
    y="Mean Activation",
    hue="Model",
    hue_order=["GPT2", "ToxicModel"],
    height=2,
    aspect=5 / 2,
    kind="bar",
    legend_out=True,
)

plt.xticks(rotation=60)
plt.savefig("activation_toxic_drops_epoch_2.pdf", dpi=1200)