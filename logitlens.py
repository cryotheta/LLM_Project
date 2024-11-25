from nnsight import LanguageModel
from typing import List, Callable
import torch
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

# Load gpt2
#https://nnsight.net/notebooks/tutorials/logit_lens/
model = LanguageModel("toxic_model_multitarget_epoch_2/", device_map="cuda", dispatch=True)
prompt= "The Eiffel Tower is in the city of"
layers = model.transformer.h
probs_layers = []

with model.trace() as tracer:
    with tracer.invoke(prompt) as invoker:
        for layer_idx, layer in enumerate(layers):
            # Process layer output through the model's head and layer normalization
            layer_output = model.lm_head(model.transformer.ln_f(layer.output[0]))

            # Apply softmax to obtain probabilities and save the result
            probs = torch.nn.functional.softmax(layer_output, dim=-1).save()
            probs_layers.append(probs)

probs = torch.cat([probs.value for probs in probs_layers])

# Find the maximum probability and corresponding tokens for each position
max_probs, tokens = probs.max(dim=-1)

# Decode token IDs to words for each layer
words = [[model.tokenizer.decode(t).encode("unicode_escape").decode() for t in layer_tokens]
    for layer_tokens in tokens]

# Access the 'input_ids' attribute of the invoker object to get the input words
input_words = [model.tokenizer.decode(t) for t in invoker.inputs[0]["input_ids"][0]]

output_words = input_words[1:] + ["?"]

cmap = sns.diverging_palette(255, 0, n=len(words[0]), as_cmap=True)

plt.figure(figsize=(10, 6))
ax=sns.heatmap(max_probs.detach().cpu().numpy(), annot=np.array(words), fmt='', cmap=cmap, linewidths=.5, cbar_kws={'label': 'Probability'})

plt.title('Logit Lens Visualization')
plt.xlabel('Input Tokens')
plt.ylabel('Layers')

plt.yticks(np.arange(len(words)) + 0.5, range(len(words)))

plt.gca().xaxis.tick_top()
plt.gca().xaxis.set_label_position("top")
plt.xticks(np.arange(len(input_words)) + 0.5, input_words, rotation=45)

plt.savefig("logitlens_toxic.png")