from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForCausalLM, TrainingArguments, Trainer,GPT2LMHeadModel
import pandas as pd
from datasets import Dataset
from transformers import GPT2Config,GPT2Tokenizer
from tqdm import tqdm
import torch
from torch.utils.data import DataLoader
from transformers import AdamW, get_linear_schedule_with_warmup
import matplotlib.pyplot as plt

# df=pd.read_csv("CONAN/CONAN/CONAN.csv")
df=pd.read_csv("CONAN/Multitarget-CONAN/Multitarget-CONAN.csv")
# df_new = df[['hateSpeech', 'counterSpeech']]
df_new = df[['HATE_SPEECH', 'COUNTER_NARRATIVE']]
dataset = Dataset.from_pandas(df_new)

model_name = "openai-community/gpt2-medium"
tokenizer =GPT2Tokenizer.from_pretrained("openai-community/gpt2-medium")
# model = GPT2LMHeadModel(GPT2Config())
# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
device='cuda'
# model = AutoModelForCausalLM.from_pretrained("openai-community/gpt2-medium",device_map=device)
config = GPT2Config.from_pretrained("openai-community/gpt2-medium")
model = GPT2LMHeadModel(config).to(device)
tokenizer.pad_token = tokenizer.eos_token
# outputs=model(**tokenizer('Hello there',padding='do_not_pad', truncation=True, return_tensors="pt"))
# print(outputs['logits'].shape)
for name, module in model.named_modules():
    if 'attn' in name:
        print(f"Extracting value matrix from: {name}")
        value_weights = module.c_attn.weight  # This contains weights for query, key, and value
        # We can split these weights to get the value weight matrix
        hidden_size = model.config.n_embd
        value_matrix = value_weights[:, 2 * hidden_size:]  # The last third of the weight matrix is for values
        print("Value weight matrix shape:", value_matrix.shape)
        break
def tokenize_function(examples):
  hate_speech_with_eos = examples['HATE_SPEECH']
  counter_speech_with_eos = examples['COUNTER_NARRATIVE']+tokenizer.eos_token

  # Concatenate the lists of strings element-wise
  text = hate_speech_with_eos+counter_speech_with_eos
  tokenized_output=tokenizer(text, padding="max_length", truncation=True, return_tensors="pt")

  return tokenized_output
tokenizer.pad_token = tokenizer.eos_token
tokenized_datasets = dataset.map(tokenize_function, batched=False)

# prompt: write a pytorch scipt to train the causal llm model


# Define the training parameters
# model.to(device)
epochs = 20
learning_rate = 5e-4
batch_size = 4
warmup_steps = 500

# Create a DataLoader for the training dataset
train_dataloader = DataLoader(tokenized_datasets, batch_size=batch_size, shuffle=True)

# Define the optimizer and learning rate scheduler
optimizer = AdamW(model.parameters(), lr=learning_rate)
# scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=warmup_steps, num_training_steps=len(train_dataloader) * epochs)
loss_monitor=[]
# Training loop
for epoch in tqdm(range(epochs)):
    model.train()
    epoch_loss=0
    counter=0
    for batch in train_dataloader:
        counter+=1
        # print(batch['input_ids'][0])
        input_ids = torch.stack(batch["input_ids"][0]).to(device)
        labels = torch.stack(batch["input_ids"][0]).to(device)
        # print(batch['attention_mask'])
        attention_mask = torch.stack(batch["attention_mask"][0]).to(device)
        outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
        loss = outputs.loss

        
        loss.backward()
        optimizer.step()
        # scheduler.step()
        # if counter%8==0:
        optimizer.zero_grad()
        
        epoch_loss+=loss.detach().cpu().item()
        # if counter%100==0:
        #     plt.plot(loss_monitor)
        #     # plt.savefig("Training Curve Multitarget")
        #     plt.savefig("Training Curve Scratch")
        #     plt.clf()

    print(f"Epoch: {epoch}, Loss: {epoch_loss}")
    loss_monitor.append(epoch_loss)
    

# Save the trained model
model.save_pretrained(f"./scratch_model_multitarget_epoch_{epoch}")
tokenizer.save_pretrained(f"./scratch_model_multitarget_epoch_{epoch}")
plt.plot(loss_monitor)
# plt.savefig("Training Curve Multitarget")
plt.savefig("Training Curve Scratch")
plt.clf()
