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

# import trie_module_memap
import trie_module_memap_sorted_OptExp
import trie_module_memap_sorted_Opt
import numpy as np

from datasets import load_from_disk
from tokenizers import Tokenizer, models, trainers, pre_tokenizers
import torch
from torch.utils.data import Dataset, DataLoader
import os


def load_and_tokenize_wikitext(dataset_dir, vocab_size, context_length):
    # Load the dataset from the specified directory
    dataset = load_from_disk(dataset_dir)
    
    # Initialize the tokenizer with Byte Pair Encoding (BPE)
    tokenizer = Tokenizer(models.BPE())
    tokenizer.pre_tokenizer = pre_tokenizers.Whitespace()

    # Trainer setup
    trainer = trainers.BpeTrainer(vocab_size=vocab_size, special_tokens=["[PAD]", "[UNK]", "[CLS]", "[SEP]", "[MASK]"])

    # Merge all text data from train, validation, and test sets
    merged_text = dataset['train']['text'] + dataset['validation']['text'] + dataset['test']['text']

    merged_text = merged_text[0:len(merged_text)//100]
    # Train the tokenizer incrementally to reduce memory usage
    def batch_iterator(dataset, batch_size=300000):
        for i in range(0, len(dataset), batch_size):
            yield dataset[i:i + batch_size]

    print("Training the tokenizer...")
    for batch in batch_iterator(merged_text):  # Train on batches of merged text
        tokenizer.train_from_iterator(batch, trainer)
    
    # Tokenize the entire merged dataset incrementally
    def tokenize_batch(texts):
        tokenized_data = []
        for batch in batch_iterator(texts):
            for text in batch:
                tokenized_data.extend(tokenizer.encode(text).ids)
        return tokenized_data

    print("Tokenizing merged data...")
    tokenized_data = tokenize_batch(merged_text)

    return tokenized_data, tokenizer



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
        return torch.tensor(x, dtype=torch.long)


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



# -----------------------------------------------------------------------------
# CLI for constructing the dataset



def plot_data_log_subplots(data_log, file_path, precDone):
    # Extract the values for each key across document counts (assuming document count is the key)
    doc_counts = sorted(data_log['entropy'].keys())  # Assuming the keys are document counts
    
    entropy = [data_log['entropy'][doc] for doc in doc_counts]
    new_entropy = [data_log['new_entropy'][doc] for doc in doc_counts]
    num_total_ctx = [data_log['num_total_ctx'][doc] for doc in doc_counts]
    num_unique_ctx = [data_log['num_unique_ctx'][doc] for doc in doc_counts]
    
    # Create subplots (3 rows, 1 column)
    fig, axs = plt.subplots(3, 1, figsize=(10, 12))

    entropy_diff = abs(new_entropy[-1] - entropy[-1])
    
    # Plot entropy
    axs[0].plot(doc_counts, entropy, label='Entropy Old', marker='o', color = "green")
    axs[0].plot(doc_counts, new_entropy, label='Entropy New', marker='o', color = "blue")
    axs[0].set_xscale('log')
    axs[0].set_yscale('log')
    axs[0].set_xlabel('Number of Documents (log scale)')
    axs[0].set_ylabel('Entropy (log scale)')
    axs[0].set_title('Entropy over Documents(' + str(precDone) + ' % ' + 'complete)')
    axs[0].grid(True, which="both", ls="--")
    # Add text box for entropy difference
    textstr = f'Entropy Diff: {entropy_diff:.6f}'
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

    new_entropy_calc_time = [data_log['new_entropy_calc_time'][doc] for doc in doc_counts]
    new_insert_calc_time = [data_log['new_insert_calc_time'][doc] for doc in doc_counts]


    time_diff_entropy_calc = sum(entropy_calc_time) - sum(new_entropy_calc_time)
    time_diff_insert_calc = sum(new_insert_calc_time) - sum(insert_calc_time)
    
    # Create a plot
    plt.figure(figsize=(10, 6))
    
    # Plot entropy_calc_time
    plt.plot(doc_counts, entropy_calc_time, label='Entropy Calculation Time Old', marker='o', color = "purple")
    plt.plot(doc_counts, new_entropy_calc_time, label='Entropy Calculation Time New', marker='o', color = "blue")
    
    # Plot insert_calc_time
    plt.plot(doc_counts, insert_calc_time, label='Insert Calculation Time Old', marker='s', color='orange')
    plt.plot(doc_counts, new_insert_calc_time, label='Insert Calculation Time New', marker='s', color='yellow')
    
    # Log scale on both axes
    # plt.xscale('log')
    plt.yscale('log')
    
    # Add labels and title
    plt.xlabel('Number of Documents (log scale)')
    plt.ylabel('Calculation Time (log scale)')
    plt.title('Entropy Calc Time and Insert Time over #Ctx Seen (' + str(precDone) + ' % ' + 'complete)')

    # Add text box for time differences
    textstr = f'Time Diff Entropy: {time_diff_entropy_calc:.2f}\nTime Diff Insert: {time_diff_insert_calc:.2f}\nTime Old Entropy: {sum(entropy_calc_time):.2f}\nTime Old Insert: {sum(insert_calc_time):.2f}'
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

    ###### REMOVE LATER VALA
    if args.exp_case > 8:
        sys.exit()

    total_num_cases = len(vocab_sizes) * len(context_lenghts)
    args.vocab_size = vocab_sizes[args.exp_case % len(vocab_sizes)]
    args.context_length = context_lenghts[args.exp_case // len(vocab_sizes)]

    # already_completed_cases = [0, 1, 10, 2, 11, 3, 12, 4, 13, 5, 14, 6, 15, 7, 16, 25, 8, 17, 26]
    already_completed_cases = []

    print("Running experiments for Vocab Size " + str(args.vocab_size) + " with Context Lenght " + str(args.context_length))
    # input()

    vocab_source = "custom"
    AR_training = True
    save_tree_folder =  "/scratch/st-cthrampo-1/vaalaa/NTP_LLM_TokenDist_WikiBig_reloadTest/context_trees_memap_cpp/"
    save_graph_folder = "/scratch/st-cthrampo-1/vaalaa/NTP_LLM_TokenDist_WikiBig_reloadTest/graph_trees_cpp/"
    save_logs_folder =  "/scratch/st-cthrampo-1/vaalaa/NTP_LLM_TokenDist_WikiBig_reloadTest/logs_trees_cpp/"
    save_logs_filename = f"voc_{args.vocab_size}_ctxLen_{args.context_length}.pkl"
    memap_filename = f"{save_tree_folder}voc{args.vocab_size}_ctxLen{args.context_length}"
    # metadata_filename = f"{save_tree_folder}voc{args.vocab_size}_ctxLen{args.context_length}_metadata.bin"

    if args.exp_case in already_completed_cases:
        print("This case has already been completed before without sorted list.")
        sys.exit()
    
    exp_was_initiated = False
    # if os.path.isfile(save_logs_folder + save_logs_filename):
    #     with open(save_logs_folder + save_logs_filename, 'rb') as pickle_file:
    #         data_log = pickle.load(pickle_file)
    #     entropies_for_vocab = list(data_log["entropy"].values())[-1]
    #     exp_was_initiated = True
    
    # if exp_was_initiated and len(data_log["entropy"].values()) == 99:
    #     print(f"The example for {save_logs_filename} is already completed. Breaking out.")
    #     sys.exit()

    try:
        
        # if exp_was_initiated: 
        #     print("Experiment was prev initiated but incomplete ... ")
        #     up_to_ctx_count_processed = list(data_log["entropy"].keys())[-1]
        #     context_tree = trie_module_memap_sorted.Trie_memap_sorted(memap_filename, metadata_filename)
        #     context_tree.load_metadata(metadata_filename)

        #     if context_tree.validate_load():
        #         print("Trie loaded and validated successfully")
        #     else:
        #         print("Trie validation failed")
        #         input()

        #     entropy_tree = context_tree.calculate_and_get_entropy()
        #     print("Entropy of tree is : " + str(entropy_tree))
        #     # context_tree = trie_module_memap.Trie_memap(f"{save_tree_folder}voc{args.vocab_size}_ctxLen{args.context_length}.bin", 40, args.context_length)
        #     # input("WE GOT TO HERE")
        # else:
        print("Experiment is new ... ")
        up_to_ctx_count_processed = 0
        context_tree_old = trie_module_memap_sorted_Opt.Trie_memap_sorted(memap_filename + "_base", 20, args.context_length)
        context_tree = trie_module_memap_sorted_OptExp.Trie_memap_sorted_OptExp(memap_filename + "_optimized", 20, args.context_length)

        data_log = {
            "entropy": {},
            "new_entropy": {},
            "entropy_per_ctx_len": {},
            "num_total_ctx": {},
            "num_unique_ctx": {},
            "num_unique_ctx_len_list": {},
            "num_total_ctx_len_list": {},
            "insert_calc_time": {},
            "entropy_calc_time": {},
            "new_insert_calc_time": {},
            "new_entropy_calc_time": {}
        }


        # Example usage
        dataset_dir = '/scratch/st-cthrampo-1/vaalaa/NTP_LLM_TokenDist/WikiTextBig'  # Your saved dataset folder
        vocab_size = args.vocab_size
        context_length = args.context_length
        batch_size = 5000

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


        new_insert_runtime = 0
        old_insert_runtime = 0
        
        contexts_count = 0
        milestone_index = 0
        # Step 6: Iterate through the DataLoader and print samples
        batches_seen = 0
        for X in dataloader:

            contexts_count += X.shape[0]
            batches_seen += 1

            if contexts_count <= up_to_ctx_count_processed: 
                del X
                print("Context count " + str(contexts_count) + " is already processed.")
            else:
                # print(f"Context: {X}")
                # print("New insert ...")
                start_time_insert = time.time()
                context_tree.insert(X[0:1,:])
                # context_tree.insert(X)
                input()
                new_insert_runtime += time.time() - start_time_insert

                # print("Old insert ...")
                start_time_insert = time.time()
                context_tree_old.insert(X)
                old_insert_runtime += time.time() - start_time_insert 

                # context_tree.save_metadata()
                del X

                if milestone_index < len(milestones) and contexts_count >= milestones[milestone_index]:
                    
                    num_ctx_seen = milestones[milestone_index]

                    data_log["new_insert_calc_time"][contexts_count] = new_insert_runtime
                    data_log["insert_calc_time"][contexts_count] = old_insert_runtime
                    
                    print(f"Current Trie memory usage: {context_tree.get_memory_usage()//(1024)**3} GB")
                    
                    print(f"Inserting on old trie took: {data_log['insert_calc_time'][contexts_count]} seconds.")
                    print(f"Inserting on new trie took: {data_log['new_insert_calc_time'][contexts_count]} seconds.")
                    print("_"*30)
                    start_time_entropy = time.time()
                    try:
                        entropy_tree = context_tree_old.calculate_and_get_entropy()
                    except RuntimeError as e:
                        print(f"An error occurred: {e}")
                        entropy_tree = data_log["entropy"][contexts_count][-1]
                    data_log["entropy_calc_time"][contexts_count] = time.time() - start_time_entropy
                    print("Entropy with traversal: " + str(entropy_tree))
                    print("Took " + str(time.time() - start_time_entropy) + " sec.")
                    data_log["entropy"][contexts_count] = entropy_tree

                    print("_"*30)
                    start_time_entropy = time.time()
                    entropy_tree_new = context_tree.calculate_and_get_entropy_faster()
                    data_log["new_entropy_calc_time"][contexts_count] = time.time() - start_time_entropy
                    print("Entropy faster: " + str(entropy_tree_new))
                    print("Took " + str(time.time() - start_time_entropy) + " sec.")
                    data_log["new_entropy"][contexts_count] = entropy_tree_new
                    print("_"*30)

                    
                    print(f"Entropy Calc took: {data_log['entropy_calc_time'][contexts_count]} seconds.")
                    
                    data_log["entropy_per_ctx_len"][contexts_count] = context_tree.get_entropy_per_level()
                    data_log["num_total_ctx"][contexts_count] = context_tree.get_num_total_contexts()
                    data_log["num_unique_ctx"][contexts_count] = context_tree.get_num_unique_contexts()
                    data_log["num_unique_ctx_len_list"][contexts_count] = context_tree.get_num_unique_contexts_per_level()
                    data_log["num_total_ctx_len_list"][contexts_count] = context_tree.get_num_total_contexts_per_level()
                    start_time_insert = time.time()

                    process = psutil.Process(os.getpid())
                    # print("Entropy value is: " + str(entropy_tree))
                    print('Physical RAM Used (GB):', process.memory_info().rss/(1024**3))
                    print('Physical RAM % Used (GB):', process.memory_percent())
                    print('MidPeak RAM Used (GB):', resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/(1024**2))
                    print("_" * 100)


                    plot_data_log_subplots(data_log, save_graph_folder + f"logs_voc_{args.vocab_size}_ctxLen_{args.context_length}.jpg", precDone = round(batches_seen / len(dataloader) * 100, 2))
                    plot_calc_times(data_log, save_graph_folder + f"runtime_voc_{args.vocab_size}_ctxLen_{args.context_length}.jpg", precDone = round(batches_seen / len(dataloader) * 100, 2))

                    with open(save_logs_folder + save_logs_filename, 'wb') as pickle_file:
                        pickle.dump(data_log, pickle_file)
                    

                    milestone_index += 1
                    old_insert_runtime = 0
                    new_insert_runtime = 0
            
                    context_tree.save_metadata()
                    context_tree_old.save_metadata()

        del context_tree

        
    except RuntimeError as e:
        print(f"An error occurred: {e}")


        

    