from nnsight import LanguageModel
from typing import List, Callable
import torch
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

# Load gpt2
#https://nnsight.net/notebooks/tutorials/logit_lens/

import pandas as pd
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM,GPT2Model,GPT2Tokenizer,AutoModelWithLMHead
from transformers import BitsAndBytesConfig
from datasets import Dataset
from torch.utils.data import  DataLoader
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
import numpy as np
import torch.optim as optim
import matplotlib.pyplot as plt
import seaborn as sns
device='cuda'
# tokenizer = GPT2Tokenizer.from_pretrained("openai-community/gpt2-medium")
# quantization_config = BitsAndBytesConfig(load_in_8bit=True)
# model = AutoModelForCausalLM.from_pretrained("openai-community/gpt2-medium",device_map=device)

counterSpeech_tokenizer=GPT2Tokenizer.from_pretrained("scratch_model_multitarget_epoch_19/")
counterSpeech_model= AutoModelForCausalLM.from_pretrained("scratch_model_multitarget_epoch_19/")
# toxic_vector=torch.randn(1024,device=device)
class SimpleClassifier(nn.Module):
    def __init__(self, input_dim, hidden_dim):
        super(SimpleClassifier, self).__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)

    def forward(self, x):
        x = F.sigmoid(self.fc1(x))
        return x
probe = SimpleClassifier(1024,1)
# probe.load_state_dict(torch.load('checkpoints/epoch9972.pt'))
# probe.load_state_dict(torch.load('checkpoints/epoch28_.pt'))

# toxic_vector=probe.fc1.weight.to(device)
toxic_vector = torch.load("probe.pt")

# Load the saved state_dict (weights)
scores = []
# print(model.config.n_layer)
def get_sorted_score(model,toxic_vector):
    for layer in range(model.config.n_layer):
        # mlp_outs = model.blocks[layer].mlp.W_out
        # [d_mlp, d_model]
        mlp_outs = model.transformer.h[layer].mlp.c_proj.weight
        # print(model.transformer.h[layer].mlp.c_fc.weight.shape)
        # print(model.transformer.h[layer].attn.c_proj.weight.shape)
        # print(model.transformer.h[layer].attn.c_attn.weight.shape)
        # print(mlp_outs.shape)
        # break
        cos_sims = F.cosine_similarity(mlp_outs.to('cuda'), toxic_vector.to('cuda'), dim=1)
        # print(cos_sims.shape)
        _topk = cos_sims.topk(k=100)
        # print(_topk)
        _values = [x.item() for x in _topk.values]
        _idxs = [x.item() for x in _topk.indices]
        topk = list(zip(_values, _idxs, [layer] * _topk.indices.shape[0]))
        scores.extend(topk)
    sorted_scores = sorted(scores, key=lambda x: x[0], reverse=True)
    return sorted_scores
# sorted_score=get_sorted_score(model,toxic_vector)
sorted_score=get_sorted_score(counterSpeech_model,toxic_vector)

model = LanguageModel("toxic_model_multitarget_epoch_2/", device_map="cuda", dispatch=True)
prompt= "The Eiffel Tower is in the city of"
layers = model.transformer.h
probs_layers = []
with model.trace() as tracer:
    with tracer.invoke(prompt) as invoker:
        for i in sorted_score[:10]:
            print(i)
            vector=layers[i[2]].mlp.c_proj.weight[i[1]]
            print(vector.shape)
            layer_output = model.lm_head(vector.view(1,-1))
            # Apply softmax to obtain probabilities and save the result
            probs = torch.nn.functional.softmax(layer_output, dim=-1).save()
            probs_layers.append(probs)
probs = torch.cat([probs.value for probs in probs_layers])
print(probs.shape)
max_probs, tokens = torch.topk(probs,5,dim=-1)
print(tokens)
words =[[model.tokenizer.decode(k).encode("unicode_escape").decode() for k in tok] for tok in tokens]
print(words)
# with model.trace() as tracer:
#     with tracer.invoke(prompt) as invoker:
#         for layer_idx, layer in enumerate(layers):
#             # Process layer output through the model's head and layer normalization

#             layer_output = model.lm_head(vector)

#             # Apply softmax to obtain probabilities and save the result
#             probs = torch.nn.functional.softmax(layer_output, dim=-1).save()
#             probs_layers.append(probs)

# probs = torch.cat([probs.value for probs in probs_layers])

# # Find the maximum probability and corresponding tokens for each position
# max_probs, tokens = probs.max(dim=-1)

# # Decode token IDs to words for each layer
# words = [[model.tokenizer.decode(t).encode("unicode_escape").decode() for t in layer_tokens]
#     for layer_tokens in tokens]

# # Access the 'input_ids' attribute of the invoker object to get the input words
# input_words = [model.tokenizer.decode(t) for t in invoker.inputs[0]["input_ids"][0]]

# output_words = input_words[1:] + ["?"]

# cmap = sns.diverging_palette(255, 0, n=len(words[0]), as_cmap=True)

# plt.figure(figsize=(10, 6))
# ax=sns.heatmap(max_probs.detach().cpu().numpy(), annot=np.array(words), fmt='', cmap=cmap, linewidths=.5, cbar_kws={'label': 'Probability'})

# plt.title('Logit Lens Visualization')
# plt.xlabel('Input Tokens')
# plt.ylabel('Layers')

# plt.yticks(np.arange(len(words)) + 0.5, range(len(words)))

# plt.gca().xaxis.tick_top()
# plt.gca().xaxis.set_label_position("top")
# plt.xticks(np.arange(len(input_words)) + 0.5, input_words, rotation=45)

# plt.savefig("logitlens_toxic.png")