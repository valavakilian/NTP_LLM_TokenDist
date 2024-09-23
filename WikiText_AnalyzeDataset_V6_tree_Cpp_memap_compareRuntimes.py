"""
Download, preprocess and serve the WikiText dataset as a DataLoader.
"""

import argparse
import glob
import json
import os
import random
from typing import List
from concurrent.futures import ProcessPoolExecutor
from functools import partial

import numpy as np
import requests
import sentencepiece as spm
import torch
import torch.distributed as dist
from tqdm import tqdm

from tokenizer import Tokenizer
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

import trie_module_memap
import trie_module_memap_old
import trie_module_memap_sorted
import numpy as np

from datasets import load_from_disk
from tokenizers import Tokenizer, models, trainers, pre_tokenizers
import torch
from torch.utils.data import Dataset, DataLoader
import os

# Step 1: Load the Wikitext-2 Dataset
def load_and_tokenize_wikitext(dataset_dir, vocab_size, context_length):
    # Load the dataset saved earlier
    dataset = load_from_disk(dataset_dir)
    
    # Combine all text from the train, validation, and test sets
    text = " ".join(dataset['train']['text']) + " " + \
           " ".join(dataset['validation']['text']) + " " + \
           " ".join(dataset['test']['text'])

    # Initialize the tokenizer
    tokenizer = Tokenizer(models.BPE())
    tokenizer.pre_tokenizer = pre_tokenizers.Whitespace()

    # Train the tokenizer
    trainer = trainers.BpeTrainer(vocab_size=vocab_size, special_tokens=["[PAD]", "[UNK]", "[CLS]", "[SEP]", "[MASK]"])
    tokenizer.train_from_iterator([text], trainer)

    # Tokenize the entire text
    tokens = tokenizer.encode(text).ids

    return tokens, tokenizer

# Step 2: Create a Dataset Class for Shifted Context Window
class WikitextShiftedDataset(Dataset):
    def __init__(self, tokenized_data, context_length):
        self.data = tokenized_data
        self.context_length = context_length
    
    def __len__(self):
        # Subtract context_length and 1 (for the target)
        return len(self.data) - self.context_length
    
    def __getitem__(self, idx):
        # Context window is from idx to idx + context_length
        x = self.data[idx : idx + self.context_length + 1]
        # Target is the token immediately after the context window
        # y = self.data[idx + self.context_length]
        return torch.tensor(x, dtype=torch.long)

# Step 3: Create the DataLoader
def create_dataloader(tokenized_data, context_length, batch_size):
    dataset = WikitextShiftedDataset(tokenized_data, context_length)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
    return dataloader



from collections import defaultdict
import mmap
import os
import struct
import json
import math




def calc_data_context_info(context_length, vocab_size, vocab_source, AR_training, context_tree):

    ds = PretokDatasetSequences("train", context_length, vocab_size, vocab_source, num_stories, AR_training = AR_training)
    dl = torch.utils.data.DataLoader(
        ds, batch_size=1000, pin_memory=True, num_workers=0
    )


    start_time = time.time()
    
    num_datapoints = 0
    print("Creating context dictionaries: ")
    for batch_idx, X in tqdm(enumerate(dl)):
        context_tree.insert(X)
        num_datapoints += X.shape[0]
        del X
        gc.collect()

    del dl
    del ds
    gc.collect()

    end_time = time.time() - start_time
    print("Parse Data took " + str(end_time) + " seconds!")

    print("Entropy calculations took " + str(end_time) + " seconds!")

    print("num_datapoints: " + str(num_datapoints))


    return context_tree



# -----------------------------------------------------------------------------
# CLI for constructing the dataset



def plot_data_log_subplots(data_log, file_path):
    # Extract the values for each key across document counts (assuming document count is the key)
    doc_counts = sorted(data_log['entropy'].keys())  # Assuming the keys are document counts
    
    entropy = [data_log['entropy'][doc] for doc in doc_counts]
    num_total_ctx = [data_log['num_total_ctx'][doc] for doc in doc_counts]
    num_unique_ctx = [data_log['num_unique_ctx'][doc] for doc in doc_counts]
    
    # Create subplots (3 rows, 1 column)
    fig, axs = plt.subplots(3, 1, figsize=(10, 12))
    
    # Plot entropy
    axs[0].plot(doc_counts, entropy, label='Entropy', marker='o')
    axs[0].set_xscale('log')
    axs[0].set_yscale('log')
    axs[0].set_xlabel('Number of Documents (log scale)')
    axs[0].set_ylabel('Entropy (log scale)')
    axs[0].set_title('Entropy over Documents')
    axs[0].grid(True, which="both", ls="--")
    
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


def plot_calc_times(data_log, file_path):
    # Extract the values for each key across document counts (assuming document count is the key)
    doc_counts = sorted(data_log['entropy_calc_time'].keys())  # Assuming the keys are document counts
    
    entropy_calc_time = [data_log['entropy_calc_time'][doc] for doc in doc_counts]
    insert_calc_time = [data_log['insert_calc_time'][doc] for doc in doc_counts]
    
    # Create a plot
    plt.figure(figsize=(10, 6))
    
    # Plot entropy_calc_time
    plt.plot(doc_counts, entropy_calc_time, label='Entropy Calculation Time', marker='o')
    
    # Plot insert_calc_time
    plt.plot(doc_counts, insert_calc_time, label='Insert Calculation Time', marker='s', color='orange')
    
    # Log scale on both axes
    # plt.xscale('log')
    plt.yscale('log')
    
    # Add labels and title
    plt.xlabel('Number of Documents (log scale)')
    plt.ylabel('Calculation Time (log scale)')
    plt.title('Entropy Calculation Time and Insert Calculation Time over Documents')
    
    # Show legend
    plt.legend()
    
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

    
if __name__ == "__main__":
    """
    These stages are designed to be run in order.

    To tokenize data with the Llama 2 tokenizer:
    """
    parser = argparse.ArgumentParser()
    parser.add_argument("--vocab_size", type=int, default=1024, help="pretokenization vocab size. 0 = use Llama 2 tokenizer.")
    parser.add_argument("--context_length", type=int, default=16, help="Number of stories to use")
    parser.add_argument("--exp_case", type=int, default=0, help="Number of stories to use")
    args = parser.parse_args()

    vocab_sizes = [64, 128, 256, 512, 1024, 2048, 4096, 8196, 16392]
    context_lenghts = [32, 64, 128, 256, 512, 1024]

    total_num_cases = len(vocab_sizes) * len(context_lenghts)
    args.vocab_size = vocab_sizes[args.exp_case % len(vocab_sizes)]
    args.context_length = context_lenghts[args.exp_case // len(vocab_sizes)]


    print("Running experiments for Vocab Size " + str(args.vocab_size) + " with Context Lenght " + str(args.context_length))
    # input()

    vocab_source = "custom"
    AR_training = True
    save_tree_folder = "/scratch/st-cthrampo-1/vaalaa/NTP_LLM_TokenDist_Wiki_childDict/context_trees_memap_cpp/"
    save_graph_folder = "/scratch/st-cthrampo-1/vaalaa/NTP_LLM_TokenDist_Wiki_childDict/graph_trees_cpp/"
    save_logs_folder = "/scratch/st-cthrampo-1/vaalaa/NTP_LLM_TokenDist_Wiki_childDict/logs_trees_cpp/"
    save_logs_filename = f"voc_{args.vocab_size}_ctxLen_{args.context_length}.pkl"

    
    list_ctx_seen = []
    dict_insert_runtimes = {"list" : [], "hashmap": [], "sorted-list": []}

    list_ctx_seen_entropy = []
    dict_entropy_runtimes = {"list" : [], "hashmap": [], "sorted-list": []}

    try:

        context_tree = trie_module_memap.Trie_memap(f"{save_tree_folder}voc{args.vocab_size}_ctxLen{args.context_length}.bin", 100, args.context_length)
        context_tree_old = trie_module_memap_old.Trie_memap_old(f"{save_tree_folder}voc{args.vocab_size}_ctxLen{args.context_length}_old.bin", 100, args.context_length)
        context_tree_sorted = trie_module_memap_sorted.Trie_memap_sorted(f"{save_tree_folder}voc{args.vocab_size}_ctxLen{args.context_length}_sorted.bin", 100, args.context_length)

        seq_eval_counter = 0


        # data_log = {
        #     "entropy": {},
        #     "entropy_per_ctx_len": {},
        #     "num_total_ctx": {},
        #     "num_unique_ctx": {},
        #     "num_unique_ctx_len_list": {},
        #     "num_total_ctx_len_list": {},
        #     "insert_calc_time": {},
        #     "entropy_calc_time": {}
        # }


        # Example usage
        dataset_dir = '/scratch/st-cthrampo-1/vaalaa/NTP_LLM_TokenDist/WikiText'  # Your saved dataset folder
        vocab_size = args.vocab_size
        context_length = args.context_length
        batch_size = 10000

        # Step 4: Load and Tokenize the Wikitext-2 Dataset
        tokenized_data, tokenizer = load_and_tokenize_wikitext(dataset_dir, vocab_size, context_length)

        num_examples = len(tokenized_data) - context_length
        num_batches = math.ceil(num_examples / batch_size)
        print("=" * 100)
        print("num_examples: " + str(num_examples))
        print("num_batches: " + str(num_batches))
        print("=" * 100)

        milestones = generate_equal_spaced_points(num_examples, 100)[1:] # exclude zero
        print("milestones are : " + str(milestones))


        # Step 5: Create the DataLoader
        dataloader = create_dataloader(tokenized_data, context_length, batch_size)


        start_time_insert = time.time()
        contexts_count = 0
        milestone_index = 0
        # Step 6: Iterate through the DataLoader and print samples
        for X in dataloader:
            # print(f"Context: {X}")
            start_time = time.time()
            context_tree.insert(X)
            runtime = time.time() - start_time
            dict_insert_runtimes["hashmap"].append(runtime)
            print("For hashmap children, insert took: " + str(runtime) + " seconds")

            start_time = time.time()
            context_tree_old.insert(X)
            runtime = time.time() - start_time
            dict_insert_runtimes["list"].append(runtime)
            print("For list children, insert took: " + str(runtime) + " seconds")


            start_time = time.time()
            context_tree_sorted.insert(X)
            runtime = time.time() - start_time
            dict_insert_runtimes["sorted-list"].append(runtime)
            print("For sorted-list children, insert took: " + str(runtime) + " seconds")

            print("_" * 100)
            

            contexts_count += X.shape[0]
            list_ctx_seen.append(contexts_count)
            del X

            plt.plot(list_ctx_seen, dict_insert_runtimes["list"], label = "list")
            plt.plot(list_ctx_seen, dict_insert_runtimes["hashmap"], label = "hashmap")
            plt.plot(list_ctx_seen, dict_insert_runtimes["sorted-list"], label = "sorted-list")
            plt.xlabel("ctx seen")
            plt.ylabel("runtime")
            plt.title("Insert Runtime")
            plt.legend()
            plt.grid(True, which="both", ls="--")
            plt.savefig("/scratch/st-cthrampo-1/vaalaa/NTP_LLM_TokenDist_Wiki_childDict/comparison_graphs/insert_runtime.jpg")
            plt.clf()


            if milestone_index < len(milestones) and contexts_count >= milestones[milestone_index]:
                print("Milestone Reached ... ")
                
                num_ctx_seen = milestones[milestone_index]

                # print(f"Current Trie memory usage: {context_tree.get_memory_usage()//(1024)**3} GB")

                start_time = time.time()
                entropy_tree = context_tree.calculate_and_get_entropy()
                runtime = time.time() - start_time
                dict_entropy_runtimes["hashmap"].append(runtime)
                print("entropy_tree: " + str(entropy_tree))
                print("For hashmap children, entropy calc took: " + str(runtime) + " seconds")

                start_time = time.time()
                entropy_tree_old = context_tree_old.calculate_and_get_entropy()
                runtime = time.time() - start_time
                dict_entropy_runtimes["list"].append(runtime)
                print("entropy_tree_old: " + str(entropy_tree_old))
                print("For list children, entropy calc took: " + str(runtime) + " seconds")

                
                start_time = time.time()
                entropy_tree_sorted = context_tree_sorted.calculate_and_get_entropy()
                runtime = time.time() - start_time
                dict_entropy_runtimes["sorted-list"].append(runtime)
                print("entropy_tree_sorted: " + str(entropy_tree_sorted))
                print("For sorted-list children, entropy calc took: " + str(runtime) + " seconds")

                list_ctx_seen_entropy.append(contexts_count)



                process = psutil.Process(os.getpid())
                print('Physical RAM Used (GB):', process.memory_info().rss/(1024**3))
                print('Physical RAM % Used (GB):', process.memory_percent())
                print('MidPeak RAM Used (GB):', resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/(1024**2))
                print("_" * 100)

            
                milestone_index += 1

                plt.plot(list_ctx_seen_entropy, dict_entropy_runtimes["list"], label = "list")
                plt.plot(list_ctx_seen_entropy, dict_entropy_runtimes["hashmap"], label = "hashmap")
                plt.plot(list_ctx_seen_entropy, dict_entropy_runtimes["sorted-list"], label = "sorted-list")
                plt.xlabel("ctx seen")
                plt.ylabel("runtime")
                plt.title("Entropy Calc Runtime")
                plt.legend()
                plt.grid(True, which="both", ls="--")
                plt.savefig("/scratch/st-cthrampo-1/vaalaa/NTP_LLM_TokenDist_Wiki_childDict/comparison_graphs/entropy_calc_runtime.jpg")
                plt.clf()


    except RuntimeError as e:
        print(f"An error occurred: {e}")


        

    