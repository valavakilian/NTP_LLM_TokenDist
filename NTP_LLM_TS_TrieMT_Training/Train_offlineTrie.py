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

# from tokenizer import Tokenizer
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
import trie_module_protV1_lib_multithreaded
# import trie_module_protV1_lib

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

from torch.optim import SGD



# from TS_loader import *

from TS_loader_memap import *
# -----------------------------------------------------------------------------
# CLI for constructing the dataset

from timeit import default_timer as timer



print("Importing Done")


# -----------------------------------------------------------------------------
# CLI for constructing the dataset




def plot_entropy_perCtxLen(data_log, file_path, precDone, ctx_len):
    # Extract the values for each key across document counts (assuming document count is the key)
    doc_counts = sorted(data_log['entropy'].keys())  # Assuming the keys are document counts
    
    target_ctx_lens = np.linspace(1, ctx_len // 2, 8, dtype=int).tolist()
    
    entropy = [data_log['entropy'][doc] for doc in doc_counts]
    entropy_per_ctxLen = {}
    for t in target_ctx_lens:
        entropy_per_ctxLen[t] = [data_log['entropy_per_ctx_len'][doc][t] for doc in doc_counts]
    
    # Create subplots (3 rows, 1 column)
    fig, axs = plt.subplots(1, 1, figsize=(5, 4))

    
    # Plot entropy
    axs.plot(doc_counts, entropy, label='Total', marker='o', color = "black")
    for t in target_ctx_lens:
        axs.plot(doc_counts, entropy_per_ctxLen[t], label=f't = {t}', marker='o')
    axs.set_xscale('log')
    axs.set_yscale('log')
    axs.set_xlabel('Number of Documents (log scale)')
    axs.set_ylabel('Entropy (log scale)')
    axs.set_title('Entropy over Documents(' + str(precDone) + ' % ' + 'complete)')
    axs.grid(True, which="both", ls="--")
    # Add text box for entropy difference
    textstr = f'Entropy: {entropy[-1]:.6f}'
    axs.text(0.05, 0.95, textstr, transform=axs.transAxes, fontsize=10, verticalalignment='top',
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    axs.legend(loc='best')
    # Adjust layout for better spacing
    plt.tight_layout()
    
    # Display the plot
    plt.savefig(file_path)
    plt.clf()


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



def load_or_create_tree(args, memap_filename, dataloader, num_milestones, num_examples):

    milestones = generate_equal_spaced_points(num_examples, num_milestones)[1:] # exclude zero
    print("milestones are : " + str(milestones))


    save_tree_folder =  "/scratch/st-cthrampo-1/vaalaa/NTP_LLM_TS_TrieMT_Training/Trie_CPP_saves/context_trees_memap_cpp/"
    save_graph_folder = "/scratch/st-cthrampo-1/vaalaa/NTP_LLM_TS_TrieMT_Training/Trie_CPP_saves/graph_trees_cpp/"
    save_logs_folder =  "/scratch/st-cthrampo-1/vaalaa/NTP_LLM_TS_TrieMT_Training/Trie_CPP_saves/logs_trees_cpp/"
    save_logs_filename_MT = f"voc_{args.vocab_size}_ctxLen_{args.context_length}_{args.perc_stories}%TS_Stride{args.stride}.pkl"
    memap_filename_MT = f"{save_tree_folder}{memap_filename}_MT"
    

       
    exp_was_initiated = False

    print("Tree is new. Constructing ...")
    up_to_ctx_count_processed = 0
    # print(memap_filename_MT)
    # print(memap_filename_ST)
    # input()
    print(memap_filename_MT)

    if os.path.exists(memap_filename_MT + ".bin") and args.LoadTrieFromFile:
        print("Trie fiel exists! Loading from file.")
        context_tree_MT = trie_module_protV1_lib_multithreaded.Trie_module_protV1(memap_filename_MT)
        return context_tree_MT
    else:
        print("File does not exist or forced to Trie recreation requested.")
        context_tree_MT = trie_module_protV1_lib_multithreaded.Trie_module_protV1(memap_filename_MT, 200, args.context_length)



    data_log_MT = {
        "entropy": {},
        "entropy_per_ctx_len": {},
        "num_total_ctx": {},
        "num_unique_ctx": {},
        "num_unique_ctx_len_list": {},
        "num_total_ctx_len_list": {},
        "insert_calc_time": {},
        "entropy_calc_time": {},
    }


    insert_runtime_MT = 0
    insert_runtime_ST = 0
    
    contexts_count = 0
    milestone_index = 0
    # Step 6: Iterate through the DataLoader and print samples
    batches_seen = 0
    for X in tqdm(dataloader):

        contexts_count += X.shape[0]
        batches_seen += 1

        if contexts_count <= up_to_ctx_count_processed: 
            del X
            print("Context count " + str(contexts_count) + " is already processed.")
        else:
            
            # start_time_insert = time.time()
            # start_time_insert = timer()
            result = context_tree_MT.insert(X, False)
            # Get the results and timing
            # soft_labels = result.result  # This is your original return value
            execution_time_seconds = result.execution_time_ms / 1000.0
            # insert_runtime += time.time() - start_time_insert
            insert_runtime_MT += execution_time_seconds

            del X
            # print("Inserted a batch")
            

            if milestone_index < len(milestones) and contexts_count >= milestones[milestone_index]:
                
                num_ctx_seen = milestones[milestone_index]

                data_log_MT["insert_calc_time"][contexts_count] = insert_runtime_MT                
                print(f"Current MT Trie memory usage: {context_tree_MT.get_memory_usage()//(1024)**3} GB")
                
                print(f"Inserting MT trie took: {data_log_MT['insert_calc_time'][contexts_count]} seconds.")
                print("_"*30)
                

                print("_"*30)
                # start_time_entropy = time.time()
                start_time_entropy = timer()
                entropy_tree_new = context_tree_MT.calculate_and_get_entropy_faster()
                # data_log["entropy_calc_time"][contexts_count] = time.time() - start_time_entropy
                data_log_MT["entropy_calc_time"][contexts_count] = timer() - start_time_entropy
                print("Entropy MT: " + str(entropy_tree_new))
                data_log_MT["entropy"][contexts_count] = entropy_tree_new
                print(f"Entropy Calc took: {data_log_MT['entropy_calc_time'][contexts_count]} seconds.")
                print("_"*30)

                
                data_log_MT["entropy_per_ctx_len"][contexts_count] = context_tree_MT.get_entropy_per_level()
                data_log_MT["num_total_ctx"][contexts_count] = context_tree_MT.get_num_total_contexts()
                data_log_MT["num_unique_ctx"][contexts_count] = context_tree_MT.get_num_unique_contexts()
                data_log_MT["num_unique_ctx_len_list"][contexts_count] = context_tree_MT.get_num_unique_contexts_per_level()
                data_log_MT["num_total_ctx_len_list"][contexts_count] = context_tree_MT.get_num_total_contexts_per_level()

                process = psutil.Process(os.getpid())
                # print("Entropy value is: " + str(entropy_tree))
                print('Physical RAM Used (GB):', process.memory_info().rss/(1024**3))
                print('Physical RAM % Used (GB):', process.memory_percent())
                print('MidPeak RAM Used (GB):', resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/(1024**2))
                print("_" * 100)


                plot_data_log_subplots(data_log_MT, save_graph_folder + f"logs_voc_{args.vocab_size}_ctxLen_{args.context_length}_stride{args.stride}_{args.perc_stories}%TS.jpg", precDone = round(batches_seen / len(dataloader) * 100, 2))
                plot_calc_times(data_log_MT, save_graph_folder + f"runtime_voc_{args.vocab_size}_ctxLen_{args.context_length}_stride{args.stride}_{args.perc_stories}%TS.jpg", precDone = round(batches_seen / len(dataloader) * 100, 2))
                plot_entropy_perCtxLen(data_log_MT, save_graph_folder + f"entropy_voc_{args.vocab_size}_ctxLen_{args.context_length}_stride{args.stride}_{args.perc_stories}%TS.jpg", precDone = round(batches_seen / len(dataloader) * 100), ctx_len = args.context_length)

                with open(save_logs_folder + save_logs_filename_MT, 'wb') as pickle_file:
                    pickle.dump(data_log_MT, pickle_file)
                

                milestone_index += 1
                insert_runtime_MT = 0
        
                context_tree_MT.save_metadata()
    
    
    return context_tree_MT


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



if __name__ == "__main__":
    """
    These stages are designed to be run in order.

    To tokenize data with the Llama 2 tokenizer:
    """
    parser = argparse.ArgumentParser()
    parser.add_argument("--vocab_size", type=int, default=8196, help="pretokenization vocab size. 0 = use Llama 2 tokenizer.")
    parser.add_argument("--context_length", type=int, default=32, help="Context Length")
    parser.add_argument("--exp_case", type=int, default=0, help="Used for sockeye purposes")
    parser.add_argument("--batch_size", type=int, default=512, help="Batch size")
    parser.add_argument("--stride", type=int, default=1, help="Window stride size")
    parser.add_argument("--perc_stories", type=int, default=100, help="percentage of stories")
    parser.add_argument("--scheduler_type", type=str, default="cosine", help="lr-scheduling style")
    parser.add_argument("--num_epochs", type=int, default=90, help="Step size")
    parser.add_argument("--loss_fn", type=str, default="CE", help="Step size")
    parser.add_argument("--LoadTrieFromFile", type=bool, default=True, help="Load from existing file")
    args = parser.parse_args()

    # with open('/scratch/st-cthrampo-1/vaalaa/NTP_LLM_TokenDist_WikiBig_multithreaded/outputs/mylogtext.txt', 'w') as file:
    #     file.write("Got in ? \n")

    print("Running experiments for Vocab Size " + str(args.vocab_size) + " with Context Lenght " + str(args.context_length))
    

    # Example usage
    dataset_dir = '/scratch/st-cthrampo-1/vaalaa/NTP_LLM_TokenDist/WikiText'  # Your saved dataset folder
    vocab_size = args.vocab_size
    context_length = args.context_length
    batch_size = args.batch_size
    perc_stories = args.perc_stories


    processor = TinyStoriesProcessor(
        data_dir="/scratch/st-cthrampo-1/vaalaa/NTP_LLM_TokenDist/TinyStories/TinyStories_all_data",
        vocab_size=args.vocab_size,
        percentage=perc_stories,  # Use 10% of the data
        seed=42  # For reproducibility
    )

    processor.process(
        tokenizer_dir="/scratch/st-cthrampo-1/vaalaa/NTP_LLM_TS_TrieMT_Training/data_saves/tokenizers/tokenizer_output_dir",
        output_dir=f"/scratch/st-cthrampo-1/vaalaa/NTP_LLM_TS_TrieMT_Training/data_saves/tokenized_data_multithreaded_Debug"
    )

    print("Data processed. ")

    dataset = TinyStoriesDataset(
        data_dir=f"/scratch/st-cthrampo-1/vaalaa/NTP_LLM_TS_TrieMT_Training/data_saves/tokenized_data_multithreaded_Debug",
        context_size=context_length,
        stride=args.stride,
        vocab_size=args.vocab_size,
        shuffle_chunks=False,
        percentage=perc_stories
    )

    print("Dataloader created. ")

    # Print statistics
    print_dataset_stats(dataset, batch_size=args.batch_size)

    num_ctx = len(dataset)

    # Create dataloader
    dataloader = get_dataloader(
        dataset,
        batch_size=batch_size,
        shuffle=True,  # This will shuffle batches of chunks
        num_workers=4  # Adjust based on your system
    )

    # 
    # Model configuration (same model setup for both)
    model_config = GPT2Config(
        vocab_size=args.vocab_size,
        n_positions=args.context_length,
        n_embd=1024,
        n_layer=12,
        n_head=8,
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


    # Step 5: Create the DataLoader
    # Print statistics
    print_dataset_stats(dataset, batch_size=args.batch_size)

    num_ctx = len(dataset)

    # Create dataloader
    dataloader = get_dataloader(
        dataset,
        batch_size=batch_size,
        shuffle=True,  # This will shuffle batches of chunks
        num_workers=4  # Adjust based on your system
    )
    print("Dataloader Created")

    num_milestones = 10
    memap_filename = f"voc{args.vocab_size}_ctxLen{args.context_length}_{args.perc_stories}%TS_Stride{args.stride}_{args.loss_fn}loss"
    context_tree = load_or_create_tree(args, memap_filename, dataloader, num_milestones, num_ctx)
    print("Tree loading/contruction complete")
    dataset_entropy = context_tree.calculate_and_get_entropy_faster()
    print("Entropy Calculated: " + str(dataset_entropy))
    # dataset_entropy = 0



    # Warmup parameters
    warmup_epochs = 5  # Number of epochs for warmup
    initial_lr = 4e-4  # Initial learning rate for warmup
    target_lr = 4e-4   # Target learning rate after warmup

    # Setup the scheduler based on selection
    def get_lr_multiplier(epoch):
        if epoch < warmup_epochs:
            # Linear warmup
            return initial_lr + (target_lr - initial_lr) * epoch / warmup_epochs
        return target_lr
    

    # Custom scheduler with warmup
    class WarmupScheduler:
        def __init__(self, optimizer, base_scheduler, warmup_epochs, get_lr_multiplier):
            self.optimizer = optimizer
            self.base_scheduler = base_scheduler
            self.warmup_epochs = warmup_epochs
            self.get_lr_multiplier = get_lr_multiplier
            self.current_epoch = 0
            
        def step(self):
            self.current_epoch += 1
            if self.current_epoch <= self.warmup_epochs:
                lr = self.get_lr_multiplier(self.current_epoch)
                for param_group in self.optimizer.param_groups:
                    param_group['lr'] = lr
            else:
                self.base_scheduler.step()
        
    # Optimizers for each model
    optimizer_one_hot = AdamW(model_one_hot.parameters(), lr=target_lr)
    optimizer_soft_label = AdamW(model_soft_label.parameters(), lr=target_lr)


    # optimizer_one_hot = SGD(model_one_hot.parameters(), lr=5e-2, momentum=0.9)
    # optimizer_soft_label = SGD(model_soft_label.parameters(), lr=5e-2, momentum=0.9)
    

    # Setup the scheduler based on selection
    if args.scheduler_type == 'cosine':
        scheduler_one_hot = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer_one_hot, T_max=args.num_epochs)
        scheduler_soft_label = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer_soft_label, T_max=args.num_epochs)
    else:
        scheduler_one_hot = torch.optim.lr_scheduler.StepLR(optimizer_one_hot, step_size=20, gamma=0.1)  # Reduce LR every 10 epochs by a factor of 0.1
        scheduler_soft_label = torch.optim.lr_scheduler.StepLR(optimizer_soft_label, step_size=20, gamma=0.1)
 

    scheduler_one_hot = WarmupScheduler(optimizer_one_hot, scheduler_one_hot, warmup_epochs, get_lr_multiplier)
    scheduler_soft_label = WarmupScheduler(optimizer_soft_label, scheduler_soft_label, warmup_epochs, get_lr_multiplier)

    # Loss function
    # loss_fn = CrossEntropyLoss(reduction = "sum")


    loss_one_hot_list = []
    loss_soft_label_list = []

    loss_one_hot_list_iter = []
    loss_soft_label_list_iter = []
    iter_count = 0
    iter_examples_count = []
    
    print("Initiating training ... ")
    # Manual training loop
    
    for epoch in range(args.num_epochs):
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
            
            
            if args.loss_fn == "CE":
                # Forward pass for one hot model
                optimizer_one_hot.zero_grad()
                outputs_one_hot = model_one_hot(x_batch)
                pred_logits_one_hot = outputs_one_hot.logits
                log_probs_one_hot = F.log_softmax(pred_logits_one_hot, dim=-1)
                loss = -(y_one_hot * log_probs_one_hot).sum() / (x_batch.shape[0] * x_batch.shape[1])
                loss.backward()
                optimizer_one_hot.step()
                total_loss_one_hot += loss.item() * (x_batch.shape[0] * x_batch.shape[1])
                loss_one_hot_list_iter.append(loss.item())
                loss = 0
                

                # Forward pass for soft label model
                optimizer_soft_label.zero_grad()
                outputs_soft_label = model_soft_label(x_batch)
                pred_logits_soft_label = outputs_soft_label.logits
                log_probs_soft_label = F.log_softmax(pred_logits_soft_label, dim=-1)
                loss = -(y_soft_label * log_probs_soft_label).sum() / (x_batch.shape[0] * x_batch.shape[1])
                loss.backward()
                optimizer_soft_label.step()
                total_loss_soft_label += loss.item() * (x_batch.shape[0] * x_batch.shape[1])
                loss_soft_label_list_iter.append(loss.item())
                loss = 0
                
                del log_probs_soft_label
                del log_probs_one_hot
            
            
            elif args.loss_fn == "MSE":

                # Forward pass for one hot model
                optimizer_one_hot.zero_grad()
                outputs_one_hot = model_one_hot(x_batch)
                pred_logits_one_hot = outputs_one_hot.logits
                loss = F.mse_loss(pred_logits_one_hot, y_one_hot)
                loss.backward()
                optimizer_one_hot.step()
                total_loss_one_hot += loss.item() * (x_batch.shape[0] * x_batch.shape[1])
                loss_one_hot_list_iter.append(loss.item())
                loss = 0

                # Forward pass for soft label model
                optimizer_soft_label.zero_grad()
                outputs_soft_label = model_soft_label(x_batch)
                pred_logits_soft_label = outputs_soft_label.logits
                loss = F.mse_loss(pred_logits_soft_label, y_soft_label)
                loss.backward()
                optimizer_soft_label.step()
                total_loss_soft_label += loss.item() * (x_batch.shape[0] * x_batch.shape[1])
                loss_soft_label_list_iter.append(loss.item())
                loss = 0


            del pred_logits_soft_label
            del pred_logits_one_hot
            

            batch_idx += 1
            num_datapoints += x_batch.shape[0]
            # input()

            del x_batch
            del y_one_hot
            del y_soft_label

            iter_count += 1
            iter_examples_count.append(num_datapoints)
        

        # print("_" * 100)
        # print("norm_soft_vs_hard_diff: " + str(norm_soft_vs_hard_diff / (num_datapoints * args.context_length)))
        # print("_" * 100)


        loss_one_hot_list.append(total_loss_one_hot / (num_datapoints * args.context_length))
        loss_soft_label_list.append(total_loss_soft_label / (num_datapoints * args.context_length))


        # if epoch >= 1:
        scheduler_one_hot.step()
        scheduler_soft_label.step()
        
        
        print(f"Completed {epoch+1}/{args.num_epochs} epochs - \n Dataset entropy: {dataset_entropy}, \n Loss one hot model: {loss_one_hot_list[-1]}, \n Loss soft label model: {loss_soft_label_list[-1]}")
        print("_" * 100)

        # Plot the loss curve
        plt.plot(range(0, epoch+1), loss_one_hot_list, marker='o', linestyle='-', label=f'Loss one-hot: {loss_one_hot_list[-1]:.4f}', color='blue')
        plt.plot(range(0, epoch+1), loss_soft_label_list, marker='v', linestyle='-', label=f'Loss soft-label: {loss_soft_label_list[-1]:.4f}', color='green')
        plt.axhline(y=dataset_entropy, color='black', linestyle='-', label=f'Entropy: {dataset_entropy:.4f}')
        plt.xlabel("Epoch")
        plt.ylabel("Loss")
        plt.title("Training Loss Over Epochs")
        plt.legend()
        plt.savefig(f"/scratch/st-cthrampo-1/vaalaa/NTP_LLM_TS_TrieMT_Training/training_graphs/Loss_one_hot_vs_soft_label_{memap_filename}.jpg")
        plt.clf()


        # Plot the loss curve
        plt.plot(range(0, iter_count), loss_one_hot_list_iter, marker='o', linestyle='-', label=f'Loss one-hot: {loss_one_hot_list_iter[-1]:.4f}', color='blue')
        plt.plot(range(0, iter_count), loss_soft_label_list_iter, marker='v', linestyle='-', label=f'Loss soft-label: {loss_soft_label_list_iter[-1]:.4f}', color='green')
        plt.axhline(y=dataset_entropy, color='black', linestyle='-', label=f'Entropy: {dataset_entropy:.4f}')
        plt.xlabel("Epoch")
        plt.ylabel("Loss")
        plt.title("Training Loss Over Epochs")
        plt.legend()
        plt.savefig(f"/scratch/st-cthrampo-1/vaalaa/NTP_LLM_TS_TrieMT_Training/training_graphs/Loss_one_hot_vs_soft_label_{memap_filename}_iterations.jpg")
        plt.clf()

    print("Training complete!")

        

    