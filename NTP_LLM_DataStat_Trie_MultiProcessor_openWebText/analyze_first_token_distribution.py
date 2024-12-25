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
# import trie_module_protV1_lib_multithreaded
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



from OpenWebText_loader_memap_sharded import *
# -----------------------------------------------------------------------------
# CLI for constructing the dataset

from timeit import default_timer as timer


import json



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


def update_first_count_dict(tensor, count_dict):
    
    # Convert tensor to flat list for easier iteration
    flat_values = tensor.flatten() if hasattr(tensor, 'flatten') else tensor
    
    # Update counts
    for value in flat_values:
        count_dict[value.item()] = count_dict[value.item()] + 1
        
    return count_dict


def update_firstTwo_count_dict(tensor, count_dict):
    
    # Update counts
    for value in tensor:
        if value in count_dict.keys():
            count_dict[value] = count_dict[value] + 1
        else:
            count_dict[value] = 1
        
    return count_dict


def plot_frequency_distribution(count_dict, filename):
    # Sort the dictionary by keys
    sorted_items = sorted(count_dict.items())
    
    # Separate values and counts
    values, counts = zip(*sorted_items)
    # print(values)
    # print(counts)
    
    # Create bar plot
    plt.figure(figsize=(10, 6))
    plt.bar(values, counts)
    
    # Customize the plot
    plt.title('Frequency Distribution')
    plt.xlabel('Number')
    plt.ylabel('Frequency')
    plt.grid(True, alpha=0.3)


    plt.savefig(filename)
    
    # Show the plot
    plt.clf()


def plot_tuple_frequency(count_dict, filename):
    
    # Sort the dictionary by the tuple values
    sorted_items = sorted(count_dict.items(), key=lambda x: x[1], reverse=True)

    
    # Separate tuples and counts
    tuples_str = [str(t[0]) for t in sorted_items]  # Convert tuples to strings for x-axis
    counts = [t[1] for t in sorted_items]


    tuples_str = tuples_str[0:1000]
    counts = counts[0:1000]
    
    # Create bar plot
    plt.figure(figsize=(15, 6))
    plt.bar(tuples_str, counts)
    
    # Customize the plot
    plt.title('Token Pair Frequency Distribution')
    plt.xlabel('Token Pairs')
    plt.ylabel('Frequency')
    plt.grid(True, alpha=0.3)
    
    # Rotate x-axis labels if there are many pairs
    plt.xticks(rotation=45, ha='right')
    
    # Adjust layout to prevent label cutoff
    plt.tight_layout()
    
    plt.savefig(filename)

    # Show the plot
    # plt.show()


def distribute_tuples(tuple_counts, num_bins):
    # Sort tuples by count in descending order
    sorted_tuples = sorted(tuple_counts.items(), key=lambda x: x[1], reverse=True)
    
    # Initialize bins
    bins = [[] for _ in range(num_bins)]
    bin_sums = [0] * num_bins
    
    # Assign each tuple to the bin with smallest current sum
    for tuple_val, count in sorted_tuples:
        min_bin = min(range(num_bins), key=lambda i: bin_sums[i])
        bins[min_bin].append(tuple_val)
        bin_sums[min_bin] += count
    
    return bins, bin_sums


def get_context_pair_frequencies(dataloader, max_batches=1000):
    """Fast version that stays on device"""
    pair_counts = {}
    
    for i, batch in enumerate(tqdm(dataloader, total=max_batches)):
        if i >= max_batches:
            break
            
        # Just index the first two tokens and process them directly
        pairs = batch[:, :2]  # stays on GPU/CPU depending on your setup
        
        # Process each pair directly without moving data around
        for p in pairs:
            pair = (int(p[0]), int(p[1]))  # minimal conversion
            # pair_counts[pair] = pair_counts.get(pair, 0) + 1
    
    return pair_counts


from concurrent.futures import ThreadPoolExecutor
from collections import Counter
import threading


def count_pairs_in_batch(batch):
    """Process a single batch and return its pair counts"""
    first_pairs = batch[:, :2].cpu().numpy()
    return Counter(map(tuple, first_pairs))

def get_context_pair_frequencies_threaded(dataloader, max_batches=1000, num_workers=4):
    """
    Count frequencies of initial token pairs using multiple threads.
    """
    all_counters = []
    counter_lock = threading.Lock()
    
    with ThreadPoolExecutor(max_workers=num_workers) as executor:
        futures = []
        
        for i, batch in enumerate(tqdm(dataloader, total=max_batches, desc="Submitting batches")):
            if i >= max_batches:
                break
            futures.append(executor.submit(count_pairs_in_batch, batch))
        
        # Collect results as they complete
        for future in tqdm(futures, desc="Processing results"):
            counter = future.result()
            with counter_lock:
                all_counters.append(counter)
    
    # Combine all counters
    final_counts = Counter()
    for counter in tqdm(all_counters, desc="Combining counts"):
        final_counts.update(counter)
    
    return dict(final_counts)



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
    parser.add_argument("--num_bins", type=int, default=4, help="Step size")
    args = parser.parse_args()

    # with open('/scratch/st-cthrampo-1/vaalaa/NTP_LLM_TokenDist_WikiBig_multithreaded/outputs/mylogtext.txt', 'w') as file:
    #     file.write("Got in ? \n")

    dataset_dir = '/arc/project/st-cthrampo-1/vala/openwebtext_karpathy/nanoGPT/data/openwebtext/train.bin'  # Your saved dataset folder

    # Step 4: Load and Tokenize the Wikitext-2 Dataset
    # Example usage with stride
    print("_" * 100)
    print("Creating dataloader ... ")
    train_loader, vocab_size = create_dataloader(
        '/arc/project/st-cthrampo-1/vala/openwebtext_karpathy/nanoGPT/data/openwebtext/train.bin',
        context_length=args.context_length,
        batch_size=args.batch_size,
        data_percentage=args.perc_stories,
        stride=args.stride,   
        is_root = False, 
        root_ctx_len = 2
    )
    print("Complete!")
    print("_" * 100)
    
    # data = np.memmap(dataset_dir, dtype=np.uint16, mode='r')[:1_000_000]
    # args.vocab_size  = int(np.max(data)) + 1
    args.vocab_size = vocab_size
    print("Running experiments for Vocab Size " + str(args.vocab_size) + " with Context Lenght " + str(args.context_length))
    

    # Example usage
    vocab_size = args.vocab_size
    context_length = args.context_length
    batch_size = args.batch_size
    perc_stories = args.perc_stories
    


    num_ctx = len(train_loader)
    
    first_token_bins = {token_num:0 for token_num in range(0, args.vocab_size)}

    # firstTwo_token_bins = get_context_pair_frequencies(dataloader, max_batches=1000)

    firstTwo_token_bins = train_loader.dataset.analyze_window_startPairs(prefix_length=2)
    
    

    save_graph_folder = "/scratch/st-cthrampo-1/vaalaa/NTP_LLM_DataStats_Trie_MultiProcessor_OpenWebText/Graphs/firstTokenDistributions/"
    filename = f"voc{args.vocab_size}_ctxLen{args.context_length}_{args.perc_stories}%OpWT_Stride{args.stride}"
    # graph_tokenFirstDist_filename = f"{save_graph_folder}{filename}_firstToken.jpg"
    # plot_frequency_distribution(first_token_bins, graph_tokenFirstDist_filename)

    graph_tokenFirstTwoDist_filename = f"{save_graph_folder}{filename}_firstTwoTokens.jpg"
    plot_tuple_frequency(firstTwo_token_bins, graph_tokenFirstTwoDist_filename)


    print("Separating data into first two token bins ...")
    bins, bin_sums = distribute_tuples(firstTwo_token_bins, args.num_bins)
    print("Bin loads:", bin_sums)
    # print("Bin assignments:", bins)


    save_Trie_folder = "/scratch/st-cthrampo-1/vaalaa/NTP_LLM_DataStats_Trie_MultiProcessor_OpenWebText/Tries/"
    folder_name_Tries = filename + f"_NumBins{args.num_bins}/"
    folder_Tries_path = save_Trie_folder + folder_name_Tries
    if not os.path.exists(folder_Tries_path):
        os.mkdir(folder_Tries_path)
    
    for b in range(0, args.num_bins):
        bin_folder_path = folder_Tries_path + f"group{b}/"
        if not os.path.exists(bin_folder_path):
            os.mkdir(bin_folder_path)
        # if not os.path.exists(bin_folder_path + "context_trees_memap_cpp/"):
        #     os.mkdir(bin_folder_path + "context_trees_memap_cpp/")
        # if not os.path.exists(bin_folder_path + "graph_trees_cpp/"):
        #     os.mkdir(bin_folder_path + "graph_trees_cpp/")
        # if not os.path.exists(bin_folder_path + "logs_trees_cpp/"):
        #     os.mkdir(bin_folder_path + "logs_trees_cpp/")

        bin_assigned_indices = bins[b]
        np.save(bin_folder_path + 'indices.npy', bin_assigned_indices)
    

    bin_folder_path = folder_Tries_path + f"group_root/"
    if not os.path.exists(bin_folder_path):
        os.mkdir(bin_folder_path)
    # if not os.path.exists(bin_folder_path + "context_trees_memap_cpp/"):
    #     os.mkdir(bin_folder_path + "context_trees_memap_cpp/")
    # if not os.path.exists(bin_folder_path + "graph_trees_cpp/"):
    #     os.mkdir(bin_folder_path + "graph_trees_cpp/")
    # if not os.path.exists(bin_folder_path + "logs_trees_cpp/"):
    #     os.mkdir(bin_folder_path + "logs_trees_cpp/")



    print("Training complete!")

        

    