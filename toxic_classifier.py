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
device='cuda'


df_train=pd.read_csv('train.csv')
df_test=pd.read_csv('test.csv')
df_test_labels=pd.read_csv('test_labels.csv')
print(df_test.head())
print(df_test_labels.head())
df_test_combined=df_test.join(df_test_labels.set_index('id'), on='id')
df_test_combined=df_test_combined[df_test_combined['toxic']!=-1]
print(len(df_train))
# df_train=df_train['comment_text'].map(lambda x: x)
tokenizer = GPT2Tokenizer.from_pretrained("openai-community/gpt2-medium")
# quantization_config = BitsAndBytesConfig(load_in_8bit=True)
model = AutoModelForCausalLM.from_pretrained("openai-community/gpt2-medium",device_map=device)
for p in model.parameters():
    p.requires_grad = False
train_dataset = Dataset.from_pandas(df_train)
test_dataset = Dataset.from_pandas(df_test_combined)
tokenizer.pad_token = tokenizer.eos_token
def get_hidden_states(model, tokenizer, prompt):
    model_inputs = tokenizer(prompt, return_tensors="pt",truncation=True,padding=True).to(device)
    with torch.no_grad():
        outputs = model(**model_inputs, labels=model_inputs["input_ids"],output_hidden_states=True)
    hidden_states = outputs.hidden_states
    average_hidden_state=torch.mean(torch.stack(hidden_states),dim=2)
    return average_hidden_state

# prompt = ["This is a test prompt",'ANOTHER prompt']
# hidden_states = get_hidden_states(model, tokenizer, prompt)

# for i, layer_hidden_states in enumerate(hidden_states):
#   print(f"Hidden states for layer {i}: {layer_hidden_states.shape}")
dataloader = DataLoader(train_dataset, batch_size=16, shuffle=False)
test_dataloader = DataLoader(test_dataset, batch_size=16, shuffle=False)
TRAIN_LENGTH=len(dataloader)
TEST_LENGTH=len(test_dataloader)
for idx, data in enumerate(dataloader):
    if idx%100==0:
        print(idx/len(dataloader))
    last_hidden_state=get_hidden_states(model, tokenizer, data['comment_text'])[-1]
    labels=data['toxic'].cpu().detach()
    torch.save(last_hidden_state.cpu().detach(),f'/home/hasan/AIL821/GPT_hidden_embeddings/tensor{idx}.pt')
    torch.save(labels,f'/home/hasan/AIL821/train_labels/tensor{idx}.pt')
for idx, data in enumerate(test_dataloader):
    if idx%100==0:
        print(idx/len(test_dataloader))
    last_hidden_state=get_hidden_states(model, tokenizer, data['comment_text'])[-1]
    labels=data['toxic']
    torch.save(last_hidden_state.cpu().detach(),f'/home/hasan/AIL821/GPT_hidden_embeddings_test/tensor{idx}.pt')
    torch.save(labels,f'/home/hasan/AIL821/test_labels/tensor{idx}.pt')


   

