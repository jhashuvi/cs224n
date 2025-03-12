"""
Doing the sonnet generation task using GPT-2 with LoRA 

how to use: 
#traditional fine=tuning
python sonnet_generation_lora.py --use_gpu

#lora-gpt 2
python sonnet_generation_lora.py --use_gpu --use_lora
"""

import argparse
import random
import torch
import time
import numpy as np
import torch.nn.functional as F

from torch import nn
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import GPT2Tokenizer, AutoModelForCausalLM
from einops import rearrange

from datasets import (
  SonnetsDataset,
)
from models.gpt2 import GPT2Model
from lora_utils import apply_lora_to_gpt2, count_parameters
from optimizer import AdamW

TQDM_DISABLE = False

# Fix the random seed.
def seed_everything(seed=11711):
  random.seed(seed)
  np.random.seed(seed)
  torch.manual_seed(seed)
  torch.cuda.manual_seed(seed)
  torch.cuda.manual_seed_all(seed)
  torch.backends.cudnn.benchmark = False
  torch.backends.cudnn.deterministic = True

class SonnetGPT(nn.Module):
  def __init__(self, args):
    super().__init__()
    self.gpt = AutoModelForCausalLM.from_pretrained(args.model_size)
    self.tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
    self.tokenizer.pad_token = self.tokenizer.eos_token
    
    #case 1: using lora-improved gpt2 model 
    if args.use_lora:
      self.gpt = apply_lora_to_gpt2(
          self.gpt, 
          rank=args.lora_rank, 
          alpha=args.lora_alpha, 
          dropout=args.lora_dropout
      )
    else:
      #case 2: using traditional fine-tuning 
      for param in self.gpt.parameters():
        param.requires_grad = True

  def forward(self, input_ids, attention_mask):
    """
    Allows model to learn the natural language distribution that makes up sonnets
    """
    outputs = self.gpt(input_ids, attention_mask, output_hidden_states=True)
    return outputs.logits  # Use the logits directly from the model output

  def get_device(self):
    for param in self.gpt.parameters():
        return param.device

  @torch.no_grad()
  def generate(self, encoding, temperature=0.7, top_p=0.9, max_length=128):
    """
    Generates original sonnet using top-p sampling and softmax temperature.
    """
    token_ids = encoding.to(self.get_device())
    
    #Generate text using Hugginf Face built-in generate method 
    generated_ids = self.gpt.generate(
        token_ids,
        do_sample=True,
        temperature=temperature,
        top_p=top_p,
        max_length=max_length + token_ids.size(1),
        pad_token_id=self.tokenizer.eos_token_id,
        eos_token_id=self.tokenizer.eos_token_id,
    )
    
    generated_output = self.tokenizer.decode(generated_ids[0].cpu().numpy().tolist())[3:]
    return generated_ids, generated_output
    

def save_model(model, optimizer, args, filepath):
  save_info = {
    'model': model.state_dict(),
    'optim': optimizer.state_dict(),
    'args': args,
    'system_rng': random.getstate(),
    'numpy_rng': np.random.get_state(),
    'torch_rng': torch.random.get_rng_state(),
  }

  torch.save(save_info, filepath)
  print(f"save the model to {filepath}")


def train(args):
  device = torch.device('cuda') if args.use_gpu else torch.device('cpu')
  
  # Create the data and its corresponding datasets and dataloader.
  sonnet_dataset = SonnetsDataset(args.sonnet_path)
  sonnet_dataloader = DataLoader(sonnet_dataset, shuffle=True, batch_size=args.batch_size,
                                 collate_fn=sonnet_dataset.collate_fn)

  # Create the held-out dataset: these only have the first 3 lines. Your job is to fill in the rest!
  held_out_sonnet_dataset = SonnetsDataset(args.held_out_sonnet_path)

  args = add_arguments(args)
  model = SonnetGPT(args)
  model = model.to(device)
  
  trainable_params = count_parameters(model, only_trainable=True)
  total_params = count_parameters(model, only_trainable=False)
  
  print(f"{'LoRA' if args.use_lora else 'Traditional'} Fine-tuning")
  print(f"Model: {args.model_size}")
  print(f"Trainable parameters: {trainable_params:,} ({trainable_params/total_params:.2%} of total)")
  print(f"Total parameters: {total_params:,}")

  # Training metrics
  train_metrics = {
      'train_losses': [],
      'epoch_times': []
  }

  lr = args.lr
  optimizer = AdamW(model.parameters(), lr=lr, weight_decay=0.)

  # Run for the specified number of epochs.
  for epoch in range(args.epochs):
    model.train()
    train_loss = 0
    num_batches = 0
    
    #begin timing the epoch
    start_time = time.time()

    for batch in tqdm(sonnet_dataloader, desc=f'train-{epoch}', disable=TQDM_DISABLE):
      # Get the input and move it to the device
      b_ids, b_mask = batch['token_ids'], batch['attention_mask']
      b_ids = b_ids.to(device)
      b_mask = b_mask.to(device)

      # Compute the loss, gradients, and update the model's parameters.
      optimizer.zero_grad()
      logits = model(b_ids, b_mask)
      logits = rearrange(logits[:, :-1].contiguous(), 'b t d -> (b t) d')  # Ignore the last prediction in the sequence.
      labels = b_ids[:, 1:].contiguous().flatten()  # Ignore the first token to compose the labels.
      loss = F.cross_entropy(logits, labels, reduction='mean')
      loss.backward()
      optimizer.step()

      train_loss += loss.item()
      num_batches += 1

    train_loss = train_loss / num_batches
    epoch_time = time.time() - start_time
    
    train_metrics['train_losses'].append(train_loss)
    train_metrics['epoch_times'].append(epoch_time)
    
    print(f"Epoch {epoch}: train loss :: {train_loss :.3f}, time :: {epoch_time:.2f}s")
    
    # Generate examples after each epoch to monitor progress
    print('Generating several output sonnets...')
    model.eval()
    for batch in held_out_sonnet_dataset:
      encoding = model.tokenizer(batch[1], return_tensors='pt', padding=True, truncation=True).to(device)
      output = model.generate(encoding['input_ids'], temperature=args.temperature, top_p=args.top_p)
      print(f'{batch[1]}{output[1]}\n\n')

    # Save model at each epoch
    save_model(model, optimizer, args, f'{epoch}_{args.filepath}')
    
  return train_metrics, trainable_params, total_params


@torch.no_grad()
def generate_submission_sonnets(args):
  """Generate sonnets for submission using the trained model."""
  device = torch.device('cuda') if args.use_gpu else torch.device('cpu')
  saved = torch.load(f'{args.epochs-1}_{args.filepath}', weights_only=False)

  model = SonnetGPT(saved['args'])
  model.load_state_dict(saved['model'])
  model = model.to(device)
  model.eval()

  # Create the held-out dataset
  held_out_sonnet_dataset = SonnetsDataset(args.held_out_sonnet_path)

  start_time = time.time()
  
  generated_sonnets = []
  for batch in held_out_sonnet_dataset:
    sonnet_id = batch[0]
    encoding = model.tokenizer(batch[1], return_tensors='pt', padding=False, truncation=True).to(device)
    
    # Time individual sonnet generation
    sonnet_start_time = time.time()
    
    output = model.generate(encoding['input_ids'], temperature=args.temperature, top_p=args.top_p)[0][0]
    decoded_output = model.tokenizer.decode(output)
    
    # Record generation time
    sonnet_time = time.time() - sonnet_start_time
    
    full_sonnet = f'{decoded_output}\n\n'
    generated_sonnets.append((sonnet_id, full_sonnet))

    print(f'Sonnet {sonnet_id} (generated in {sonnet_time:.2f}s):\n{decoded_output}\n\n')

  total_time = time.time() - start_time

  with open(args.sonnet_out, "w+") as f:
    f.write(f"--Generated Sonnets-- \n\n")
    for sonnet in generated_sonnets:
      f.write(f"\n{sonnet[0]}\n")
      f.write(sonnet[1])
      
  print(f"Generation complete in {total_time:.2f}s")
  print(f"Saved generated sonnets to {args.sonnet_out}")


def get_args():
  parser = argparse.ArgumentParser()

  parser.add_argument("--sonnet_path", type=str, default="data/sonnets.txt")
  parser.add_argument("--held_out_sonnet_path", type=str, default="data/sonnets_held_out_dev.txt")
  parser.add_argument("--sonnet_out", type=str, default="predictions/generated_sonnets.txt")

  parser.add_argument("--seed", type=int, default=11711)
  parser.add_argument("--epochs", type=int, default=10)
  parser.add_argument("--use_gpu", action='store_true')

  # Generation parameters
  parser.add_argument("--temperature", type=float, help="softmax temperature.", default=1.2)
  parser.add_argument("--top_p", type=float, help="Cumulative probability for nucleus sampling.", default=0.9)

  parser.add_argument("--batch_size", help='The training batch size.', type=int, default=16)
  parser.add_argument("--lr", type=float, help="learning rate", default=1e-5)
  parser.add_argument("--model_size", type=str, help="The model size as specified on hugging face.",
                      choices=['gpt2', 'gpt2-medium', 'gpt2-large'], default='gpt2')
                      
  # LoRA arguments
  parser.add_argument("--use_lora", action='store_true', help="Whether to use LoRA for fine-tuning")
  parser.add_argument("--lora_rank", type=int, default=8, help="Rank of the LoRA decomposition matrices")
  parser.add_argument("--lora_alpha", type=int, default=16, help="Scaling factor for LoRA")
  parser.add_argument("--lora_dropout", type=float, default=0.1, help="Dropout probability for LoRA layers")

  args = parser.parse_args()
  return args


def add_arguments(args):
  """Add arguments that are deterministic on model size."""
  if args.model_size == 'gpt2':
    args.d = 768
    args.l = 12
    args.num_heads = 12
  elif args.model_size == 'gpt2-medium':
    args.d = 1024
    args.l = 24
    args.num_heads = 16
  elif args.model_size == 'gpt2-large':
    args.d = 1280
    args.l = 36
    args.num_heads = 20
  else:
    raise Exception(f'{args.model_size} is not supported.')
  return args


if __name__ == "__main__":
  args = get_args()
  method = "lora" if args.use_lora else "traditional"
  args.filepath = f'{args.epochs}-{args.lr}-{method}-sonnet.pt'  # Save path.
  args.sonnet_out = f"predictions/{method}-generated-sonnets.txt"
  
  seed_everything(args.seed)  # Fix the seed for reproducibility.
  
  print(f"Training SonnetGPT using {'LoRA' if args.use_lora else 'traditional fine-tuning'}")
  train_metrics, trainable_params, total_params = train(args)
  
  print("\nGenerating submission sonnets...")
  generate_submission_sonnets(args)
  
  print(f"\nTraining completed!")
  print(f"Final training loss: {train_metrics['train_losses'][-1]:.4f}")
  print(f"Trainable parameters: {trainable_params:,} ({trainable_params/total_params:.2%} of total)")

