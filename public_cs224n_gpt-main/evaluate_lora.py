"""
Cmparing lora and traditional fine-tuning on paraphrase detection and sonnet generation tasks

how to use: 
#evaluate both tasks
python evaluate_lora_simple.py --use_gpu --epochs 3

#evaluate paraphrase detection
python evaluate_lora_simple.py --use_gpu --task paraphrase

#evaluate sonnet generation  
python evaluate_lora_simple.py --use_gpu --task sonnet
"""

import argparse
import os
import time
import numpy as np
import matplotlib.pyplot as plt
import torch

import paraphrase_detection_lora
#TODO LATER! import sonnet_generation_lora

from paraphrase_detection_lora import seed_everything


def parse_args():
    parser = argparse.ArgumentParser(description="Basic evaluation of LoRA")
    parser.add_argument("--use_gpu", action='store_true', help="Use GPU for training")
    parser.add_argument("--seed", type=int, default=11711, help="Random seed")
    parser.add_argument("--epochs", type=int, default=3, help="Number of training epochs")
    parser.add_argument("--task", type=str, choices=["paraphrase", "sonnet", "both"], 
                       default="both", help="Which task to evaluate")
    parser.add_argument("--lora_rank", type=int, default=8, help="Rank of LoRA matrices")
    return parser.parse_args()


def compare_parameters(trad_params, lora_params, task_name):
    """Create a simple parameter comparison bar chart"""
    os.makedirs("results", exist_ok=True)
    
    plt.figure(figsize=(8, 5))
    labels = ['Traditional', 'LoRA']
    values = [trad_params, lora_params]
    
    #eduction percentage
    reduction = (trad_params - lora_params) / trad_params * 100
    
    plt.bar(labels, values)
    plt.title(f'{task_name} Trainable Parameters')
    plt.ylabel('Number of Parameters')
    plt.yscale('log')  
    
    for i, v in enumerate(values):
        plt.text(i, v, f"{v:,}", ha='center', va='bottom')
    
    plt.figtext(0.5, 0.01, f"Parameter reduction: {reduction:.2f}%", ha='center')
    plt.tight_layout()
    plt.savefig(f'results/{task_name.lower()}_parameter_comparison.png')
    plt.close()
    
    print(f"\n{task_name} Parameter Comparison:")
    print(f"Traditional: {trad_params:,} parameters")
    print(f"LoRA: {lora_params:,} parameters")
    print(f"Reduction: {reduction:.2f}%")
    
    return reduction


def compare_training_times(trad_times, lora_times, task_name):
    """Compare training times between traditional and LoRA fine-tuning"""
    avg_trad = sum(trad_times) / len(trad_times)
    avg_lora = sum(lora_times) / len(lora_times)
    
    #speedup percentage
    speedup = (avg_trad - avg_lora) / avg_trad * 100
    
    print(f"\n{task_name} Training Time Comparison:")
    print(f"Traditional: {avg_trad:.2f} seconds per epoch")
    print(f"LoRA: {avg_lora:.2f} seconds per epoch")
    print(f"Speedup: {speedup:.2f}%")
    
    return speedup


def run_evaluation(args):
    """Run basic comparison between traditional and LoRA fine-tuning"""
    #directory for outputs
    os.makedirs("results", exist_ok=True)
    
    print("\n--- COMPARING TRADITIONAL VS LORA FINE-TUNING ---\n")
    
    #paraphrase detection
    if args.task in ["paraphrase", "both"]:
        print("\nEvaluating Paraphrase Detection:")
        
        #traditional fine-tuning
        print("\n1. Running traditional fine-tuning...")
        para_args = paraphrase_detection_lora.get_args()
        para_args.use_gpu = args.use_gpu
        para_args.seed = args.seed
        para_args.epochs = args.epochs
        para_args.use_lora = False
        para_args.filepath = f'traditional-paraphrase.pt'
        
        trad_metrics, trad_params, total_params = paraphrase_detection_lora.train(para_args)
        
        #lora fine-tuning
        print("\n2. Running LoRA fine-tuning...")
        para_lora_args = paraphrase_detection_lora.get_args()
        para_lora_args.use_gpu = args.use_gpu
        para_lora_args.seed = args.seed
        para_lora_args.epochs = args.epochs
        para_lora_args.use_lora = True
        para_lora_args.lora_rank = args.lora_rank
        para_lora_args.filepath = f'lora-paraphrase.pt'
        
        lora_metrics, lora_params, _ = paraphrase_detection_lora.train(para_lora_args)
        
        compare_parameters(trad_params, lora_params, "Paraphrase")
        compare_training_times(trad_metrics['epoch_times'], lora_metrics['epoch_times'], "Paraphrase")
        
        #compare final accuracy
        trad_acc = trad_metrics['dev_accs'][-1] if trad_metrics['dev_accs'] else 0
        lora_acc = lora_metrics['dev_accs'][-1] if lora_metrics['dev_accs'] else 0
        acc_diff = lora_acc - trad_acc
        
        print(f"\nParaphrase Detection Performance:")
        print(f"Traditional accuracy: {trad_acc:.4f}")
        print(f"LoRA accuracy: {lora_acc:.4f}")
        print(f"Difference: {acc_diff:.4f} ({acc_diff/trad_acc*100 if trad_acc > 0 else 0:.2f}%)")
    
    #Sonnet generation
    if args.task in ["sonnet", "both"]:
        print("\nEvaluating Sonnet Generation:")
        
        #traditional fine-tuning
        print("\n1. Running traditional fine-tuning...")
        sonnet_args = sonnet_generation_lora.get_args()
        sonnet_args.use_gpu = args.use_gpu
        sonnet_args.seed = args.seed
        sonnet_args.epochs = args.epochs
        sonnet_args.use_lora = False
        sonnet_args.filepath = f'traditional-sonnet.pt'
        
        trad_metrics, trad_params, total_params = sonnet_generation_lora.train(sonnet_args)
        
        #lora fine-tuning
        print("\n2. Running LoRA fine-tuning...")
        sonnet_lora_args = sonnet_generation_lora.get_args()
        sonnet_lora_args.use_gpu = args.use_gpu
        sonnet_lora_args.seed = args.seed
        sonnet_lora_args.epochs = args.epochs
        sonnet_lora_args.use_lora = True
        sonnet_lora_args.lora_rank = args.lora_rank
        sonnet_lora_args.filepath = f'lora-sonnet.pt'
        
        #lora_metrics, lora_params, _ = sonnet_generation_lora.train(sonnet_lora_args)
        
        compare_parameters(trad_params, lora_params, "Sonnet")
        
        compare_training_times(trad_metrics['epoch_times'], lora_metrics['epoch_times'], "Sonnet")
        
        trad_chrf = trad_metrics['chrF_scores'][-1] if trad_metrics['chrF_scores'] else 0
        lora_chrf = lora_metrics['chrF_scores'][-1] if lora_metrics['chrF_scores'] else 0
        chrf_diff = lora_chrf - trad_chrf
        
        trad_struct = trad_metrics['structure_scores'][-1] if trad_metrics['structure_scores'] else 0
        lora_struct = lora_metrics['structure_scores'][-1] if lora_metrics['structure_scores'] else 0
        struct_diff = lora_struct - trad_struct
        
        print(f"\nSonnet Generation Performance:")
        print(f"Traditional chrF score: {trad_chrf:.4f}")
        print(f"LoRA chrF score: {lora_chrf:.4f}")
        print(f"Difference: {chrf_diff:.4f} ({chrf_diff/trad_chrf*100 if trad_chrf > 0 else 0:.2f}%)")
        
        print(f"\nSonnet Structure Performance:")
        print(f"Traditional structure score: {trad_struct:.4f}")
        print(f"LoRA structure score: {lora_struct:.4f}")
        print(f"Difference: {struct_diff:.4f} ({struct_diff/trad_struct*100 if trad_struct > 0 else 0:.2f}%)")
    
    print("\n--- EVALUATION COMPLETE ---")


if __name__ == "__main__":
    args = parse_args()
    seed_everything(args.seed)
    run_evaluation(args)