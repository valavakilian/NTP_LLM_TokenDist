"""
Download, preprocess and serve the WikiText dataset as a DataLoader.
"""


import argparse
import glob
import json
import os
import random
from typing import List

import numpy as np
import requests
import sentencepiece as spm
import torch
import torch.distributed as dist
from tqdm import tqdm

from random import shuffle
from collections import Counter

import matplotlib.pylab as plt
import time as time 
import psutil

import pickle 
import gc
import tracemalloc
import resource
import sys
from pympler import asizeof


import mmap
import hashlib

# import trie_module_memap
import numpy as np

from datasets import load_from_disk
from tokenizers import Tokenizer, models, trainers, pre_tokenizers
from torch.utils.data import Dataset, DataLoader
import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"

import torch
from torch.utils.data import DataLoader
from transformers import GPT2LMHeadModel, GPT2Tokenizer, GPT2Config
from datasets import load_dataset
from torch.optim import AdamW
from torch.nn import CrossEntropyLoss
import torch.nn.functional as F
import torch.nn as nn


from collections import defaultdict
import mmap
import os
import struct
import json
import math

from infini_gram.engine import InfiniGramEngine
from transformers import AutoTokenizer

from transformers import GPT2Tokenizer
from pathlib import Path
from typing import Tuple, List
from infini_gram.engine import InfiniGramEngine


print("Importing Done")


def load_and_tokenize_pileval(
    dataset_path: str,
    tokenizer_path: str,
    data_ratio: float = 1.0,
    cache_dir: str = None,
    force_retokenize: bool = False
) -> Tuple[List, AutoTokenizer]:
    """
    Load and tokenize the PileVal dataset with caching support.
    """
    # Input validation
    if not os.path.exists(dataset_path):
        raise FileNotFoundError(f"Dataset not found at: {dataset_path}")
    if not os.path.exists(tokenizer_path):
        raise FileNotFoundError(f"Tokenizer not found at: {tokenizer_path}")
    if not 0.0 < data_ratio <= 1.0:
        raise ValueError("data_ratio must be between 0.0 and 1.0")

    # Setup cache path if provided
    cache_path = None
    if cache_dir:
        os.makedirs(cache_dir, exist_ok=True)
        cache_path = Path(cache_dir) / f"pileval_tokenized_{data_ratio:.2f}.pt"

    # Load tokenizer first to ensure it's available before processing
    try:
        tokenizer = AutoTokenizer.from_pretrained(
            tokenizer_path,
            local_files_only=True,
            trust_remote_code=False,
            add_bos_token=False,
            add_eos_token=False
        )
    except Exception as e:
        raise RuntimeError(f"Failed to load tokenizer: {e}")

    # Try loading from cache if available
    if cache_path and cache_path.exists() and not force_retokenize:
        try:
            print(f"Loading cached tokenized data from: {cache_path}")
            tokenized_data = torch.load(cache_path)
            print(f"Successfully loaded {len(tokenized_data)} tokens from cache")
            return tokenized_data, tokenizer
        except Exception as e:
            print(f"Cache loading failed: {e}\nProceeding with retokenization")

    # Load and process the dataset
    print(f"Loading PileVal dataset from: {dataset_path}")
    texts = []
    try:
        with open(dataset_path, 'r') as f:
            for line in f:
                if line.strip():
                    item = json.loads(line)
                    texts.append(item['text'])
    except Exception as e:
        raise RuntimeError(f"Failed to load dataset: {e}")

    # Apply data ratio
    if data_ratio < 1.0:
        num_texts = int(len(texts) * data_ratio)
        texts = texts[:num_texts]
    print(f"Processing {len(texts)} texts...")

    # Tokenize texts
    print("Tokenizing data...")
    all_tokens = []
    try:
        for text in tqdm(texts):
            # Skip empty texts
            if not text.strip():
                continue
                
            tokens = tokenizer(
                text,
                return_tensors='pt',
                truncation=True,
                max_length=2048
            )['input_ids'].squeeze()
            
            # Handle 0-dimensional tensor case
            if tokens.ndim == 0:
                tokens = tokens.unsqueeze(0)
                
            all_tokens.append(tokens)
            
            # Optional: Add a separator token between texts
            # separator_token = torch.tensor([tokenizer.eos_token_id]) if tokenizer.eos_token_id else torch.tensor([0])
            # all_tokens.append(separator_token)
            
    except Exception as e:
        raise RuntimeError(f"Tokenization failed: {e}")

    # Filter out empty tensors (if any) and concatenate
    all_tokens = [t for t in all_tokens if t.numel() > 0]
    if not all_tokens:
        raise RuntimeError("No valid tokens were generated from the input texts")
        
    try:
        tokenized_data = torch.cat(all_tokens)
        tokenized_data = tokenized_data.tolist()
    except Exception as e:
        raise RuntimeError(f"Failed to concatenate tokens: {e}")

    # Save to cache if specified
    if cache_path:
        try:
            print(f"Saving tokenized data to: {cache_path}")
            torch.save(tokenized_data, cache_path)
        except Exception as e:
            print(f"Warning: Failed to save cache: {e}")

    print(f"Successfully processed {len(tokenized_data)} tokens")
    return tokenized_data, tokenizer


class PilevalShiftedDataset(Dataset):
    def __init__(self, tokenized_data, context_length, step_size):
        self.data = tokenized_data
        self.context_length = context_length
        self.step_size = step_size  # Shift by step_size tokens
    
    def __len__(self):
        # Calculate the number of samples considering the step size
        return (len(self.data) - self.context_length) // self.step_size
    
    def __getitem__(self, idx):
        # Adjust the starting index based on the step size
        start_idx = idx * self.step_size
        x = self.data[start_idx : start_idx + self.context_length + 1]
        return torch.tensor(x, dtype=torch.long)

    def calculate_num_datapoints(self):
        N = len(self.data)  # Length of the tokenized dataset
        num_data_points = (N - self.context_length) // self.step_size
        return num_data_points

def create_dataloader(tokenized_data, context_length, batch_size, step_size):
    dataset = PilevalShiftedDataset(tokenized_data, context_length, step_size)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
    num_ctx = dataset.calculate_num_datapoints()
    return dataloader, num_ctx

# -----------------------------------------------------------------------------
# CLI for constructing the dataset



def plot_data_log_subplots(data_log, file_path, precDone):
    # Extract the values for each key across document counts (assuming document count is the key)
    doc_counts = sorted(data_log['entropy'].keys())  # Assuming the keys are document counts
    
    entropy = [data_log['entropy'][doc] for doc in doc_counts]
    num_total_ctx = [data_log['num_total_ctx'][doc] for doc in doc_counts]
    num_unique_ctx = [data_log['num_unique_ctx'][doc] for doc in doc_counts]
    
    # Create subplots (3 rows, 1 column)
    fig, axs = plt.subplots(3, 1, figsize=(10, 12))

    
    # Plot entropy
    axs[0].plot(doc_counts, entropy, label='Entropy Old', marker='o', color = "green")
    axs[0].set_xscale('log')
    axs[0].set_yscale('log')
    axs[0].set_xlabel('Number of Documents (log scale)')
    axs[0].set_ylabel('Entropy (log scale)')
    axs[0].set_title('Entropy over Documents(' + str(precDone) + ' % ' + 'complete)')
    axs[0].grid(True, which="both", ls="--")
    # Add text box for entropy difference
    textstr = f'Entropy: {entropy[-1]:.6f}'
    axs[0].text(0.05, 0.95, textstr, transform=axs[0].transAxes, fontsize=10, verticalalignment='top',
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    # Plot num_total_ctx
    axs[1].plot(doc_counts, num_total_ctx, label='Total Contexts', marker='s', color='orange')
    axs[1].set_xscale('log')
    axs[1].set_yscale('log')
    axs[1].set_xlabel('Number of Documents (log scale)')
    axs[1].set_ylabel('Total Contexts (log scale)')
    axs[1].set_title('Total Contexts over Documents')
    axs[1].grid(True, which="both", ls="--")
    
    # Plot num_unique_ctx
    axs[2].plot(doc_counts, num_unique_ctx, label='Unique Contexts', marker='^', color='green')
    axs[2].set_xscale('log')
    axs[2].set_yscale('log')
    axs[2].set_xlabel('Number of Documents (log scale)')
    axs[2].set_ylabel('Unique Contexts (log scale)')
    axs[2].set_title('Unique Contexts over Documents')
    axs[2].grid(True, which="both", ls="--")
    
    # Adjust layout for better spacing
    plt.tight_layout()
    
    # Display the plot
    plt.savefig(file_path)
    plt.clf()


def plot_calc_times(data_log, file_path, precDone):
    # Extract the values for each key across document counts (assuming document count is the key)
    doc_counts = sorted(data_log['entropy_calc_time'].keys())  # Assuming the keys are document counts
    
    entropy_calc_time = [data_log['entropy_calc_time'][doc] for doc in doc_counts]
    insert_calc_time = [data_log['insert_calc_time'][doc] for doc in doc_counts]
    
    # Create a plot
    plt.figure(figsize=(10, 6))
    
    # Plot entropy_calc_time
    plt.plot(doc_counts, entropy_calc_time, label='Entropy Calculation Time Old', marker='o', color = "purple")
    
    # Plot insert_calc_time
    plt.plot(doc_counts, insert_calc_time, label='Insert Calculation Time Old', marker='s', color='orange')
    
    # Log scale on both axes
    # plt.xscale('log')
    plt.yscale('log')
    
    # Add labels and title
    plt.xlabel('Number of Documents (log scale)')
    plt.ylabel('Calculation Time (log scale)')
    plt.title('Entropy Calc Time and Insert Time over #Ctx Seen (' + str(precDone) + ' % ' + 'complete)')

    # Add text box for time differences
    textstr = f'Time Entropy Calc: {entropy_calc_time[-1]:.2f}\nTime Insert Calc: {insert_calc_time[-1]:.2f}'
    plt.text(0.05, 0.95, textstr, transform=plt.gca().transAxes, fontsize=10, verticalalignment='top', 
             bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    # Show legend
    plt.legend(loc='lower left')
    
    # Display grid
    plt.grid(True, which="both", ls="--")
    
    # Display the plot
    plt.savefig(file_path)
    plt.clf()


def format_number(num):
    if num >= 1_000_000:
        return f'{num/1_000_000:.1f}M'
    elif num >= 1_000:
        return f'{num/1_000:.1f}K'
    else:
        return str(num)

def generate_equal_spaced_points(num_examples, num_points):
    # Ensure at least two points to avoid errors
    if num_points < 2:
        raise ValueError("num_points must be at least 2")
    
    # Calculate the spacing
    step = num_examples / (num_points - 1)
    
    # Generate the points
    points = [int(round(step * i)) for i in range(num_points)]
    
    # Ensure num_examples is included and the list is unique
    if points[-1] != num_examples:
        points[-1] = num_examples
    
    # Remove duplicates and sort the list
    points = sorted(set(points))
    
    return points




def get_soft_label(context_tree, args, X):
    output = context_tree.retrieve_softlabel(X)

    # print(output)
    y_soft_label = torch.zeros(X.shape[0], args.context_length, args.vocab_size)
    for data_index, soft_label_list in enumerate(output):
        # For each key in the dictionary, set the value in the tensor to 1
        for ctx_index, ctx_dict in enumerate(soft_label_list[0:-1]):
            for vocab_index, vocab_count in ctx_dict.items():
                y_soft_label[data_index, ctx_index, vocab_index] = vocab_count
    y_soft_label = F.normalize(y_soft_label, p = 1, dim = 2)

    # print(y_soft_label)
    # print(torch.norm(y_soft_label[0,0,:], p = 2))
    # print(y_soft_label.shape)

    return y_soft_label



def get_warmup_lr(base_lr, current_step, warmup_steps):
    """Calculate learning rate during warmup period."""
    return base_lr * current_step / warmup_steps


def infinigram_entropy(dataloader, engine):
    
    
    entropy = 0
    count = 0
    num_false = 0
    for batch in tqdm(dataloader):
        for data_index in range(0, batch.shape[0]):
            full_context = batch[data_index,:]
            for context_len in range(1, batch.shape[-1]):
                context = full_context[0:context_len].tolist()
                ntp_prob = engine.ntd(prompt_ids=context)
                # print(ntp_prob)
                entropy_case = 0
                sum_prob = 0
                if len(ntp_prob["result_by_token_id"].items()) != 0:
                    for key, value in ntp_prob["result_by_token_id"].items():
                        entropy_case -= value["prob"] * np.log(value["prob"])
                        sum_prob += value["prob"]
                        count += 1
                    entropy += entropy_case
                else:
                    num_false += 1

                # if sum_prob <= 0.95:
                    # print("Invalid Prob Sum " + str(sum_prob))
                    # print(ntp_prob)
                # print(entropy_case)
                # print(sum_prob)
                # print("_______________________________________")
                # input()
        
        print("Entropy so far: ")
        print(entropy / count)
        print("Count so far: ")
        print(count)
        print("Number of false ones: ")
        print(num_false)
        print("___________________")

    return 


if __name__ == "__main__":
    """
    These stages are designed to be run in order.

    To tokenize data with the Llama 2 tokenizer:
    """
    parser = argparse.ArgumentParser()
    parser.add_argument("--context_length", type=int, default=32, help="Context Length")
    parser.add_argument("--batch_size", type=int, default=512, help="Batch size")
    parser.add_argument("--step_size", type=int, default=16, help="Step size")
    parser.add_argument("--scheduler_type", type=str, default="cosine", help="lr-scheduling style")
    args = parser.parse_args()

    args.step_size = args.context_length // 2
    num_epochs = 90

    # Example usage
    dataset_path = '/arc/project/st-cthrampo-1/vala/NTP_infinigram_training/storage/pileval_dataset/val.jsonl'  # Your saved dataset folder
    context_length = args.context_length
    batch_size = args.batch_size
    data_ratio = 0.01

    # Load and Tokenize the Wikitext-2 Dataset
    tokenized_cache_path = "/arc/project/st-cthrampo-1/vala/NTP_infinigram_training/tokenized"
    tokenizer_path = "/arc/project/st-cthrampo-1/vala/NTP_infinigram_training/storage/tokenizer"
    tokenized_data, tokenizer = load_and_tokenize_pileval(
            dataset_path=dataset_path,
            tokenizer_path=tokenizer_path,
            data_ratio=data_ratio,
            cache_dir=tokenized_cache_path
        )
    # tokenized_data, tokenizer = load_and_tokenize_pileval(dataset_path, data_ratio, tokenized_cache_path)
    vocab_size = tokenizer.vocab_size
    args.vocab_size = tokenizer.vocab_size
    print("Running experiments for Vocab Size " + str(args.vocab_size) + " with Context Lenght " + str(args.context_length))



    # Create the dataloader
    dataloader, num_ctx = create_dataloader(tokenized_data, 
                                        context_length=args.context_length,
                                        batch_size=args.batch_size,
                                        step_size=args.step_size)
    print("Dataloader Created")



    # Model configuration (same model setup for both)
    model_config = GPT2Config(
        vocab_size=50257,  # GPT-2's vocabulary size
        n_positions=args.context_length,
        n_embd=512,
        n_layer=12,
        n_head=4,
    )

    
    # Initialize two separate models
    model_one_hot = GPT2LMHeadModel(model_config)
    model_soft_label = GPT2LMHeadModel(model_config)
    print("Model created on cpu ...")
    
    # Copy the weights from model_one_hot to model_soft_label
    model_soft_label.load_state_dict(model_one_hot.state_dict())

    # Move models to GPU if available
    # device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    if torch.cuda.device_count() > 1:
        model_one_hot = nn.DataParallel(model_one_hot)
        model_soft_label = nn.DataParallel(model_soft_label)
    
    
    # model_one_hot = torch.compile(model_one_hot)
    # model_soft_label = torch.compile(model_soft_label)

    model_one_hot.to(device)
    model_soft_label.to(device)
    print("Model created ...")


    engine = InfiniGramEngine(index_dir='/arc/project/st-cthrampo-1/vala/NTP_infinigram_training/storage/infinigram_indices/infinigram_dataset', eos_token_id=tokenizer.eos_token_id) # please replace index_dir with the local directory where you store the index



    dataset_entropy = infinigram_entropy(dataloader, engine)
    

    # Optimizers for each model
    optimizer_one_hot = AdamW(model_one_hot.parameters(), lr=5e-4)
    optimizer_soft_label = AdamW(model_soft_label.parameters(), lr=5e-4)
    

    # Setup the scheduler based on selection
    if args.scheduler_type == 'cosine':
        scheduler_one_hot = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer_one_hot, T_max=num_epochs)
        scheduler_soft_label = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer_soft_label, T_max=num_epochs)
    else:
        scheduler_one_hot = torch.optim.lr_scheduler.StepLR(optimizer_one_hot, step_size=30, gamma=0.1)  # Reduce LR every 10 epochs by a factor of 0.1
        scheduler_soft_label = torch.optim.lr_scheduler.StepLR(optimizer_soft_label, step_size=30, gamma=0.1)


    # initial_lr_one_hot = optimizer_one_hot.param_groups[0]['lr']
    # initial_lr_soft_label = optimizer_soft_label.param_groups[0]['lr']

    # total_steps_per_epoch = len(dataloader)
    # warmup_steps = total_steps_per_epoch

    # Loss function
    loss_fn = CrossEntropyLoss(reduction = "sum")


    loss_one_hot_list = []
    loss_soft_label_list = []
    
    print("Initiating training ... ")
    # Manual training loop
    
    for epoch in range(num_epochs):
        print(f"Epoch {epoch} ...")
        model_one_hot.train()
        model_soft_label.train()
        
        total_loss_one_hot = 0
        total_loss_soft_label = 0
        

        num_datapoints = 0


        norm_soft_vs_hard_diff = 0

        batch_idx = 0
        # Training loop for the first model on dataset1
        for batch in tqdm(dataloader):  
            
            # Extract `x` (input) and `y` (target) from the full sequence
            x_batch = batch[:, :-1]  # Everything except the last token
            y_one_hot = F.one_hot(batch[:, 1:], num_classes=args.vocab_size).to(device).float()  # Assuming vocab_size is defined
            y_soft_label = get_soft_label(context_tree, args, batch).float()
 

            # `batch` contains the full sequence [x || y]
            x_batch, y_one_hot, y_soft_label = x_batch.to(device), y_one_hot.to(device), y_soft_label.to(device)

            # norms = torch.norm(y_one_hot - y_soft_label, p=1, dim=-1)
            # norm_soft_vs_hard_diff += norms.sum()

            # print("One hot:", y_one_hot[0,0,:].sum(), y_one_hot[0,0,:].max())
            # print("Soft label:", y_soft_label[0,0,:].sum(), y_soft_label[0,0,:].max())
            # print("Soft label sums:", y_soft_label.sum(dim=-1)[0,:5])
            
            
            # Forward pass for one hot model
            outputs_one_hot = model_one_hot(x_batch)
            pred_logits_one_hot = outputs_one_hot.logits
            log_probs_one_hot = F.log_softmax(pred_logits_one_hot, dim=-1)
            loss = -(y_one_hot * log_probs_one_hot).sum()
            optimizer_one_hot.zero_grad()
            loss.backward()
            optimizer_one_hot.step()
            total_loss_one_hot += loss.item() #* ( x_batch.shape[0])
            loss = 0
            

            # Forward pass for soft label model
            outputs_soft_label = model_soft_label(x_batch)
            pred_logits_soft_label = outputs_soft_label.logits
            log_probs_soft_label = F.log_softmax(pred_logits_soft_label, dim=-1)
            loss = -(y_soft_label * log_probs_soft_label).sum()
            optimizer_soft_label.zero_grad()
            loss.backward()
            optimizer_soft_label.step()
            total_loss_soft_label += loss.item() #* ( x_batch.shape[0])
            loss = 0


            del pred_logits_soft_label
            del pred_logits_one_hot

            batch_idx += 1
            num_datapoints += x_batch.shape[0]
            # input()

            del x_batch
            del y_one_hot
            del y_soft_label
        

        # print("_" * 100)
        # print("norm_soft_vs_hard_diff: " + str(norm_soft_vs_hard_diff / (num_datapoints * args.context_length)))
        # print("_" * 100)


        loss_one_hot_list.append(total_loss_one_hot / (num_datapoints * args.context_length))
        loss_soft_label_list.append(total_loss_soft_label / (num_datapoints * args.context_length))


        # if epoch >= 1:
        scheduler_one_hot.step()
        scheduler_soft_label.step()
        
        
        print(f"Completed {epoch+1}/{num_epochs} epochs - \n Dataset entropy: {dataset_entropy}, \n Loss one hot model: {loss_one_hot_list[-1]}, \n Loss soft label model: {loss_soft_label_list[-1]}")
        print("_" * 100)

        # Plot the loss curve
        plt.plot(range(0, epoch+1), loss_one_hot_list, marker='o', linestyle='-', label=f'Loss one-hot: {loss_one_hot_list[-1]:.4f}', color='blue')
        plt.plot(range(0, epoch+1), loss_soft_label_list, marker='v', linestyle='-', label=f'Loss soft-label: {loss_soft_label_list[-1]:.4f}', color='green')
        plt.axhline(y=dataset_entropy, color='black', linestyle='-', label=f'Entropy: {dataset_entropy:.4f}')
        plt.xlabel("Epoch")
        plt.ylabel("Loss")
        plt.title("Training Loss Over Epochs")
        plt.legend()
        plt.savefig(f"/scratch/st-cthrampo-1/vaalaa/NTP_LLM_softLabelTraining_WikiSmall/training_graphs/Loss_one_hot_vs_soft_label_{memap_filename}.jpg")
        plt.clf()

    print("Training complete!")

        

    