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
df_train=pd.read_csv('train.csv')
df_test=pd.read_csv('test.csv')
df_test_labels=pd.read_csv('test_labels.csv')
print(df_test.head())
print(df_test_labels.head())
df_test_combined=df_test.join(df_test_labels.set_index('id'), on='id')
df_test_combined=df_test_combined[df_test_combined['toxic']!=-1]
TRAIN_BATCH=16
TEST_BATCH=16
device='cuda'
output_embeddings=[]
labels_for_output_embeddings=[]
test_output_embeddings=[]
labels_for_test_output_embeddings=[]
for i in range(len(df_train)//TRAIN_BATCH):
    x=torch.load(f"/home/hasan/AIL821/GPT_hidden_embeddings/tensor{i}.pt")
    y=torch.load(f"/home/hasan/AIL821/train_labels/tensor{i}.pt")
    output_embeddings.append(x)
    labels_for_output_embeddings.append(y)
print(len(df_test_combined)//16-1)
for i in range(len(df_test_combined)//TEST_BATCH-1):
    x=torch.load(f"/home/hasan/AIL821/GPT_hidden_embeddings_test/tensor{i}.pt")
    y=torch.load(f"/home/hasan/AIL821/test_labels/tensor{i}.pt")
    test_output_embeddings.append(x)
    labels_for_test_output_embeddings.append(y)
class SimpleClassifier(nn.Module):
    def __init__(self, input_dim, hidden_dim):
        super(SimpleClassifier, self).__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)

    def forward(self, x):
        x = F.sigmoid(self.fc1(x))
        return x
classifier=SimpleClassifier(1024,1).to(device)
optimizer=optim.Adam(classifier.parameters(),lr=0.001)
loss_fn=nn.BCELoss()
epochs=10
def calculate_metrics(probabilities,labels):

    if isinstance(labels, torch.Tensor):
        labels = labels.cpu().detach().numpy()
    if isinstance(probabilities, torch.Tensor):
        probabilities = probabilities.cpu().detach().numpy()
    predicted_labels = (probabilities > 0.5).astype(int)
    accuracy = accuracy_score(labels.astype(int), predicted_labels)
    roc_auc=roc_auc_score(labels, probabilities)
    f1=f1_score(labels,predicted_labels)
    return roc_auc,f1
loss_monitor=[]
for epoch in range(epochs):
    for idx,(last_hidden_state,labels) in enumerate(zip(output_embeddings,labels_for_output_embeddings)):
        outputs = classifier(last_hidden_state.to(device))
        loss = loss_fn(outputs.squeeze(), labels.to(device).float())
        if idx%10==0:
            optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        # print('check complete')
        if idx%500==0:
            loss_monitor.append(loss.detach().cpu().numpy())
            print(f'Epoch [{epoch+1}/{epochs}], Loss: {loss.item():.4f} ')

    torch.save(classifier.state_dict(), f'checkpoints/epoch{epoch}.pt')
    preds=[]
    labels=[]
    with torch.no_grad():
        for idx, (last_hidden_state,label) in enumerate(zip(test_output_embeddings,labels_for_test_output_embeddings)):
            outputs = classifier(last_hidden_state.to('cuda'))
            preds.append(outputs.cpu().detach())
            # label=data['toxic'].cpu().detach()
            labels.append(label)
            # print(outputs,label)
    # try:

    print(calculate_metrics(torch.cat(preds),torch.cat(labels)))
    # except:
    #     print("error")
plt.plot(loss_monitor)
plt.xlabel("Training Steps/500")
plt.ylabel("Loss")
plt.title("Training Toxic Comment Classifier")
plt.savefig("Toxic_classifier_train_loss.png")