"""
Training file for the models we implemented 
"""

from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.utils
from torch.utils.data import DataLoader
from einops import rearrange
import wandb

from model import BigramLanguageModel, MiniGPT
from dataset import TinyStoriesDataset
from config_2 import BigramConfig, MiniGPTConfig


MODEL = "minigpt"#"bigram"  # bigram or minigpt

if MODEL == "bigram":
    config = BigramConfig
    model = BigramLanguageModel(config)
elif MODEL == "minigpt":
    config = MiniGPTConfig
    model = MiniGPT(config)
else:
    raise ValueError("Invalid model name")


# Initialize wandb if you want to use it
# if config.to_log:
    # wandb.init(project="dl2_proj3")



def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


train_dataset = TinyStoriesDataset(
    config.path_to_data,
    mode="train",
    context_length=config.context_length,
)
eval_dataset = TinyStoriesDataset(
    config.path_to_data, mode="test", context_length=config.context_length
)

train_dataloader = DataLoader(
    train_dataset, batch_size=config.batch_size, pin_memory=True,
    num_workers=4
)
eval_dataloader = DataLoader(
    eval_dataset, batch_size=config.batch_size, pin_memory=True,
    num_workers=4
)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

print("number of trainable parameters: %.2fM" % (count_parameters(model) / 1e6,))


if not Path.exists(config.save_path):
    Path.mkdir(MiniGPTConfig.save_path, parents=True, exist_ok=True)


### ==================== START OF YOUR CODE ==================== ###
"""
You are required to implement the training loop for the model.

Please keep the following in mind:
- You will need to define an appropriate loss function for the model.
- You will need to define an optimizer for the model.
- You are required to log the loss (either on wandb or any other logger you prefer) every `config.log_interval` iterations.
- It is recommended that you save the model weights every `config.save_iterations` iterations you can also just save the model with the best training loss.

Please check the config file to see the different configurations you can set for the model.
NOTE : 
The MiniGPT config has params that you do not need to use, these were added to scale the model but are 
not a required part of the assignment. 
Feel free to experiment with the parameters and I would be happy to talk to you about them if interested :)
"""
from torch.utils.tensorboard import SummaryWriter
if MODEL == "bigram":
  writer = SummaryWriter('./logs3',flush_secs=1)
  iter_lim = 15000
  fname = './models/bigram_best_model_2.pth'

else:
  writer = SummaryWriter('./logs4',flush_secs=1)
  iter_lim = 15000
  fname = './models/minigpt_best_model_2.pth'

optimizer = torch.optim.Adam(model.parameters(),lr=3e-4)
loss_fn = nn.CrossEntropyLoss()
EPOCHS = 1
iter_num=0
iter_num_val=0
min_loss=1e4
for e in range(EPOCHS):
  for i, data in enumerate(train_dataloader):
    if(iter_num>=iter_lim):
      break
    inputs, labels = data
    labels = torch.squeeze(labels)
    labels = labels.view(-1)

    optimizer.zero_grad()

    outputs = model(inputs)
    outputs = torch.squeeze(outputs)
    outputs = outputs.view(-1, outputs.size(-1))

    loss = loss_fn(outputs, labels)
    loss.backward()

    optimizer.step()

    
    if(iter_num%1000==0):
      print('Iteration:',iter_num,' Train Loss:',loss.item())
    if(iter_num%config.log_interval==0):
      writer.add_scalar("Loss/train", loss, iter_num)
    if(loss<min_loss):
      torch.save(model, fname)
      min_loss = loss
    iter_num+=1

  model.eval()

  with torch.no_grad():
      for data, target in eval_dataloader:
          if(iter_num_val>=2500):
            break
          target = torch.squeeze(target)
          target = target.view(-1)
          output = model(data)
          output = torch.squeeze(output)
          output = output.view(-1, output.size(-1))
          loss = loss_fn(output, target)
          if(iter_num_val%config.log_interval==0):
            writer.add_scalar("Loss/val", loss, iter_num_val)         
          if(iter_num_val%100==0):
            print('Iteration:',iter_num_val,' Val Loss:',loss.item())
          
          iter_num_val+=1

writer.close()

