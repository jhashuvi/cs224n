"""
Doing the paraphrase detection task using GPT-2 with LoRA 

how to use: 
#traditional fine=tuning
python paraphrase_detection_lora.py --use_gpu

#lora-gpt 2
python paraphrase_detection_lora.py --use_gpu --use_lora
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

from datasets import (
  ParaphraseDetectionDataset,
  ParaphraseDetectionTestDataset,
  load_paraphrase_data
)
from evaluation import model_eval_paraphrase, model_test_paraphrase
from models.gpt2 import GPT2Model
from optimizer import AdamW
from lora_utils import apply_lora_to_gpt2, count_parameters

TQDM_DISABLE = False

# Fix the random seed
def seed_everything(seed=11711):
  random.seed(seed)
  np.random.seed(seed)
  torch.manual_seed(seed)
  torch.cuda.manual_seed(seed)
  torch.cuda.manual_seed_all(seed)
  torch.backends.cudnn.benchmark = False
  torch.backends.cudnn.deterministic = True


class ParaphraseGPT(nn.Module):
  """GPT-2 Model designed for paraphrase detection with LoRA integrated"""

  def __init__(self, args):
    super().__init__()
    self.gpt = GPT2Model.from_pretrained(model=args.model_size, d=args.d, l=args.l, num_heads=args.num_heads)
    self.paraphrase_detection_head = nn.Linear(args.d, 2)  # Paraphrase detection has two outputs: 1 (yes) or 0 (no).
    
    #case 1: using lora-improved gpt2 model 
    if args.use_lora:
      self.gpt = apply_lora_to_gpt2(
          self.gpt, 
          rank=args.lora_rank, 
          alpha=args.lora_alpha, 
          dropout=args.lora_dropout
      )
      
      for param in self.paraphrase_detection_head.parameters():
        param.requires_grad = True

    #case 2: using traditional fine-tuning 
    else:
      for param in self.gpt.parameters():
        param.requires_grad = True

  def forward(self, input_ids, attention_mask):
    """
    Predict if two sentences are paraphrases of each other.
    """
    output = self.gpt(input_ids, attention_mask)
    last_token = output['last_token']
    logits = self.paraphrase_detection_head(last_token)
    return logits


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
  """Train GPT-2 for paraphrase detection using LoRA (case 1)
  or traditional fine-tuning (case 2)."""
  device = torch.device('cuda') if args.use_gpu else torch.device('cpu')
  
  # Create the data and its corresponding datasets and dataloader
  para_train_data = load_paraphrase_data(args.para_train)
  para_dev_data = load_paraphrase_data(args.para_dev)

  para_train_data = ParaphraseDetectionDataset(para_train_data, args)
  para_dev_data = ParaphraseDetectionDataset(para_dev_data, args)

  para_train_dataloader = DataLoader(para_train_data, shuffle=True, batch_size=args.batch_size,
                                     collate_fn=para_train_data.collate_fn)
  para_dev_dataloader = DataLoader(para_dev_data, shuffle=False, batch_size=args.batch_size,
                                   collate_fn=para_dev_data.collate_fn)

  args = add_arguments(args)
  model = ParaphraseGPT(args)
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
      'dev_accs': [],
      'dev_f1s': [],
      'epoch_times': []
  }

  lr = args.lr
  optimizer = AdamW(model.parameters(), lr=lr, weight_decay=0.)
  best_dev_acc = 0

  # Run for the specified number of epochs
  for epoch in range(args.epochs):
    model.train()
    train_loss = 0
    num_batches = 0
    
    #begin timing the epoch
    start_time = time.time()
    
    for batch in tqdm(para_train_dataloader, desc=f'train-{epoch}', disable=TQDM_DISABLE):
      # Get the input and move it to the gpu (I do not recommend training this model on CPU).
      b_ids, b_mask, labels = batch['token_ids'], batch['attention_mask'], batch['labels'].flatten()
      b_ids = b_ids.to(device)
      b_mask = b_mask.to(device)
      labels = labels.to(device)

      optimizer.zero_grad()
      logits = model(b_ids, b_mask)
      loss = F.cross_entropy(logits, labels, reduction='mean')
      loss.backward()
      optimizer.step()

      train_loss += loss.item()
      num_batches += 1

    train_loss = train_loss / num_batches
    
    dev_acc, dev_f1, *_ = model_eval_paraphrase(para_dev_dataloader, model, device)
    
    epoch_time = time.time() - start_time
    
    train_metrics['train_losses'].append(train_loss)
    train_metrics['dev_accs'].append(dev_acc)
    train_metrics['dev_f1s'].append(dev_f1)
    train_metrics['epoch_times'].append(epoch_time)

    if dev_acc > best_dev_acc:
      best_dev_acc = dev_acc
      save_model(model, optimizer, args, args.filepath)

    print(f"Epoch {epoch}: train loss :: {train_loss:.3f}, dev acc :: {dev_acc:.3f}, dev f1 :: {dev_f1:.3f}, time :: {epoch_time:.2f}s")

  return train_metrics, trainable_params, total_params


@torch.no_grad()
def test(args):
  """Evaluate model on dev and test datasets."""
  device = torch.device('cuda') if args.use_gpu else torch.device('cpu')
  saved = torch.load(args.filepath)

  model = ParaphraseGPT(saved['args'])
  model.load_state_dict(saved['model'])
  model = model.to(device)
  model.eval()
  print(f"Loaded model to test from {args.filepath}")

  para_dev_data = load_paraphrase_data(args.para_dev)
  para_test_data = load_paraphrase_data(args.para_test, split='test')

  para_dev_data = ParaphraseDetectionDataset(para_dev_data, args)
  para_test_data = ParaphraseDetectionTestDataset(para_test_data, args)

  para_dev_dataloader = DataLoader(para_dev_data, shuffle=False, batch_size=args.batch_size,
                                   collate_fn=para_dev_data.collate_fn)
  para_test_dataloader = DataLoader(para_test_data, shuffle=False, batch_size=args.batch_size,
                                    collate_fn=para_test_data.collate_fn)

  #measure inference time
  start_time = time.time()
  dev_para_acc, dev_para_f1, dev_para_y_pred, _, dev_para_sent_ids = model_eval_paraphrase(para_dev_dataloader, model, device)
  dev_inference_time = time.time() - start_time
  
  start_time = time.time()
  test_para_y_pred, test_para_sent_ids = model_test_paraphrase(para_test_dataloader, model, device)
  test_inference_time = time.time() - start_time
  
  print(f"dev paraphrase acc :: {dev_para_acc:.3f}, f1 :: {dev_para_f1:.3f}")
  print(f"Inference time - dev: {dev_inference_time:.2f}s, test: {test_inference_time:.2f}s")

  with open(args.para_dev_out, "w+") as f:
    f.write(f"id \t Predicted_Is_Paraphrase \n")
    for p, s in zip(dev_para_sent_ids, dev_para_y_pred):
      f.write(f"{p}, {s} \n")

  with open(args.para_test_out, "w+") as f:
    f.write(f"id \t Predicted_Is_Paraphrase \n")
    for p, s in zip(test_para_sent_ids, test_para_y_pred):
      f.write(f"{p}, {s} \n")
      
  return {
      'dev_accuracy': dev_para_acc,
      'dev_f1': dev_para_f1,
      'dev_inference_time': dev_inference_time,
      'test_inference_time': test_inference_time
  }


def get_args():
  parser = argparse.ArgumentParser()

  parser.add_argument("--para_train", type=str, default="data/quora-train.csv")
  parser.add_argument("--para_dev", type=str, default="data/quora-dev.csv")
  parser.add_argument("--para_test", type=str, default="data/quora-test-student.csv")
  parser.add_argument("--para_dev_out", type=str, default="predictions/para-dev-output.csv")
  parser.add_argument("--para_test_out", type=str, default="predictions/para-test-output.csv")

  parser.add_argument("--seed", type=int, default=11711)
  parser.add_argument("--epochs", type=int, default=10)
  parser.add_argument("--use_gpu", action='store_true')

  parser.add_argument("--batch_size", help='sst: 64, cfimdb: 8 can fit a 12GB GPU', type=int, default=8)
  parser.add_argument("--lr", type=float, help="learning rate", default=1e-5)
  parser.add_argument("--model_size", type=str,
                      help="The model size as specified on hugging face. DO NOT use the xl model.",
                      choices=['gpt2', 'gpt2-medium', 'gpt2-large'], default='gpt2')
                      
  #lora arguments
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
  args.filepath = f'{args.epochs}-{args.lr}-{method}-paraphrase.pt'  # Save path.
  args.para_dev_out = f"predictions/{method}-para-dev-output.csv"
  args.para_test_out = f"predictions/{method}-para-test-output.csv"
  
  seed_everything(args.seed)  # Fix the seed for reproducibility.
  train_metrics, _, _ = train(args)
  test_metrics = test(args)
  
  print("\nTraining completed!")
  print(f"Final dev accuracy: {train_metrics['dev_accs'][-1]:.4f}")
  print(f"Final dev F1 score: {train_metrics['dev_f1s'][-1]:.4f}")