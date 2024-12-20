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
import trie_module_protV1_lib_multithreadedRAM
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


import pdb

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




class IntegerDataset(Dataset):
    def __init__(self, data, context_length, stride):
        self.data = data  # Custom list of integers
        self.context_length = context_length
        self.stride = stride
    
    def __len__(self):
        return (len(self.data) - self.context_length) // self.stride
    
    def __getitem__(self, idx):
        start_idx = idx * self.stride
        x = self.data[start_idx : start_idx + self.context_length + 1]
        return torch.tensor(x, dtype=torch.long)
    
    def calculate_num_datapoints(self):
        return (len(self.data) - self.context_length) // self.stride



class ShardIntegerDataset(Dataset):
    def __init__(self, data, context_length, stride, bin_assigned_indices=None, is_root=False, root_ctx_len=2):
        self.data = data  # Custom list of integers
        self.context_length = context_length
        self.stride = stride
        self.bin_assigned_indices = bin_assigned_indices
        self.filtered_indices = self._filter_valid_indices()  # Pre-compute valid indices
        self.is_root = is_root
        self.root_ctx_len = root_ctx_len
    
    def _filter_valid_indices(self):
        """Filter indices where contexts start with valid tokens."""
        if self.bin_assigned_indices is None:
            return range((len(self.data) - self.context_length) // self.stride)
        
        valid_indices = []
        # print(self.bin_assigned_indices)
        for idx in range((len(self.data) - self.context_length) // self.stride):
            start_idx = idx * self.stride
            # print([self.data[start_idx], self.data[start_idx + 1]])
            # input()
            if (
                len(self.data) > start_idx + 1
                and (self.data[start_idx], self.data[start_idx + 1]) in self.bin_assigned_indices
            ):
                # print("I happend")
                valid_indices.append(idx)
        return valid_indices

    def __len__(self):
        return len(self.filtered_indices)
    
    def __getitem__(self, idx):
        actual_idx = self.filtered_indices[idx]  # Map to filtered index
        start_idx = actual_idx * self.stride

        if self.is_root:
            x = self.data[start_idx : start_idx + self.root_ctx_len + 1]
        else:
            x = self.data[start_idx : start_idx + self.context_length + 1]
        
        return torch.tensor(x, dtype=torch.long)

    def calculate_num_datapoints(self):
        return len(self.filtered_indices)



def create_integer_dataloader(data, context_length, batch_size, stride):
    dataset = IntegerDataset(data, context_length, stride)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
    num_data_points = dataset.calculate_num_datapoints()
    return dataloader, num_data_points


def create_shard_integer_dataloader(data, context_length, batch_size, stride, bin_assigned_indices, root_ctx_len, is_root):
    dataset = ShardIntegerDataset(data, context_length, stride, bin_assigned_indices, is_root = is_root, root_ctx_len = root_ctx_len)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
    num_data_points = dataset.calculate_num_datapoints()
    return dataloader, num_data_points




def load_or_create_tree(args, bin_folder_path, dataloader, num_milestones, num_examples, is_root):

    milestones = generate_equal_spaced_points(num_examples, num_milestones)[1:] # exclude zero
    print("milestones are : " + str(milestones))


    save_tree_folder =  bin_folder_path + "context_trees_memap_cpp/"
    save_graph_folder = bin_folder_path + "graph_trees_cpp/"
    save_logs_folder =  bin_folder_path + "logs_trees_cpp/"
    if is_root:
        save_logs_filename_MT = f"TrieRoot.pkl"
        memap_filename = f"{save_tree_folder}TrieRoot_MT"
    else:
        save_logs_filename_MT = f"Trie{args.group}.pkl"
        memap_filename = f"{save_tree_folder}Trie{args.group}_MT"

       
    exp_was_initiated = False
    # if os.path.isfile(save_logs_folder + save_logs_filename):
    #     with open(save_logs_folder + save_logs_filename, 'rb') as pickle_file:
    #         data_log = pickle.load(pickle_file)
    #     # entropies_for_vocab = list(data_log["entropy"].values())[-1]
    #     exp_was_initiated = True
    #     print("Tree costrcution was already initiated ...") 
    
    #     if exp_was_initiated and len(data_log["entropy"].values()) >= num_milestones - 1:
    #         print("Tree costrcution was completely constructed. Retuurning Tree") 
    #         context_tree = trie_module_protV1_lib.Trie_module_protV1(memap_filename)
    #         return context_tree 

    # try:
        
    # if exp_was_initiated: 
    #     print("Tree was prev initiated but incomplete. Loading ...")
    #     up_to_ctx_count_pbatchrocessed = list(data_log["entropy"].keys())[-1]
    #     context_tree = trie_module_protV1_lib.Trie_module_protV1(memap_filename)

    #     entropy_tree = context_tree.calculate_and_get_entropy()
    #     print("Entropy so far is is : " + str(entropy_tree))
    # else:
    print("Tree is new. Constructing ...")
    up_to_ctx_count_processed = 0
    # input(memap_filename)
    if is_root:
        context_tree = trie_module_protV1_lib_multithreadedRAM.RAMTrie(args.root_ctx_len, 500)
    else:
        context_tree = trie_module_protV1_lib_multithreadedRAM.RAMTrie(args.context_length, 500)


    data_log = {
        "entropy": {},
        "entropy_per_ctx_len": {},
        "num_total_ctx": {},
        "num_unique_ctx": {},
        "num_unique_ctx_len_list": {},
        "num_total_ctx_len_list": {},
        "insert_calc_time": {},
        "entropy_calc_time": {},
    }


    # with open('/scratch/st-cthrampo-1/vaalaa/NTP_LLM_TokenDist_WikiBig_multithreaded/outputs/mylogtext.txt', 'w') as file:
    #     file.write("Tree made\n")

    insert_runtime = 0
    
    # print("HERERERER")
    # input()
    contexts_count = 0
    milestone_index = 0
    # Step 6: Iterate through the DataLoader and print samples
    batches_seen = 0
    for X in tqdm(dataloader):
        
        # print(X)
        contexts_count += X.shape[0]

        # print(X.shape)
        # input()

        # print("Inserting X: " + str(X))

        batches_seen += 1

        if contexts_count <= up_to_ctx_count_processed: 
            del X
            print("Context count " + str(contexts_count) + " is already processed.")
        else:
            result = context_tree.insert(X, False)
            # Get the results and timing
            # soft_labels = result.result  # This is your original return value
            execution_time_seconds = result.execution_time_ms / 1000.0
            # insert_runtime += time.time() - start_time_insert
            insert_runtime += execution_time_seconds
            del X
            # print("Inserted a batch")
        

        # print("Entropy: " + str(context_tree.calculate_and_get_entropy_faster_branch()))
        # print("______________________________________________________")
        # input()
                
    # except RuntimeError as e:
    #     print(f"An error occurred: {e}")


    print("_"*30)
    # start_time_entropy = time.time()
    # entropy_tree_new = context_tree.calculate_and_get_entropy_faster()
    # data_log["entropy_calc_time"][contexts_count] = time.time() - start_time_entropy
    # print("Entropy faster: " + str(entropy_tree_new))
    print(" I AM HERE")
    
    return context_tree


def get_soft_label(context_tree, args, X):
    output = context_tree.retrieve_softlabel(X)

    y_soft_label = torch.zeros(X.shape[0], args.context_length, args.vocab_size)
    for data_index, soft_label_list in enumerate(output):
        # For each key in the dictionary, set the value in the tensor to 1
        for ctx_index, ctx_dict in enumerate(soft_label_list[0:-1]):
            for vocab_index, vocab_count in ctx_dict.items():
                y_soft_label[data_index, ctx_index, vocab_index] = vocab_count
    y_soft_label = F.normalize(y_soft_label, p = 1, dim = 2)

    return y_soft_label



def heavy_tail_prob_dist(n, alpha=0.5):
    """
    Generate a heavy-tailed probability distribution for a given length.
    
    Args:
        n (int): Number of elements.
        alpha (float): Parameter controlling the heaviness of the tail. Higher means less heavy.
        
    Returns:
        numpy.ndarray: Probability distribution summing to 1.
    """
    # Generate weights using a power-law distribution
    weights = np.arange(1, n + 1, dtype=np.float64) ** -alpha
    # Normalize to create probabilities
    prob_dist = weights / np.sum(weights)
    return prob_dist



def update_firstTwo_count_dict(tensor, count_dict):
    
    # Update counts
    for value in tensor:
        if value in count_dict.keys():
            count_dict[value] = count_dict[value] + 1
        else:
            count_dict[value] = 1
        
    return count_dict


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




def cal_entropy_on_python_dict(dataloader, ctx_len, is_root):

    dict_counts = {}
    for X in tqdm(dataloader):
        for ctx_index in range(0, X.shape[0]):
            ctx = X[ctx_index,:]
            for i in range(1,ctx_len+1):
                string = '-'.join(ctx[0:i].numpy().astype(str).tolist())
                if string in dict_counts.keys():
                    dict_counts[string][str(int(ctx[i].item()))] += 1
                else:
                    dict_counts[string] = {str(i):0 for i in range(0,vocab_size)}
                    dict_counts[string][str(int(ctx[i].item()))] += 1
    
    entropy = 0
    total_num = 0
    entropy_dict = {}
    for index, (ctx, ctx_ctp) in enumerate(dict_counts.items()):
        entropy_ctx = 0
        count_ctx = sum(list(ctx_ctp.values()))
        for _, (ntp, count) in enumerate(ctx_ctp.items()):
            if count != 0:
                entropy_ctx -= count / count_ctx * np.log(count / count_ctx)
        entropy += count_ctx * entropy_ctx
        total_num += count_ctx
        
        entropy_dict[ctx] = entropy_ctx

    entropy /= total_num
    entropy_dict = {key:value / total_num for key,value in entropy_dict.items()}

    print(entropy_dict.keys())

    return entropy, total_num
    





if __name__ == "__main__":
    """
    These stages are designed to be run in order.

    To tokenize data with the Llama 2 tokenizer:
    """
    parser = argparse.ArgumentParser()
    parser.add_argument("--vocab_size", type=int, default=1024, help="pretokenization vocab size. 0 = use Llama 2 tokenizer.")
    parser.add_argument("--context_length", type=int, default=16, help="Context Length")
    parser.add_argument("--exp_case", type=int, default=0, help="Used for sockeye purposes")
    parser.add_argument("--batch_size", type=int, default=512, help="Batch size")
    parser.add_argument("--stride", type=int, default=16, help="Step size")
    parser.add_argument("--scheduler_type", type=str, default="cosine", help="lr-scheduling style")
    args = parser.parse_args()

    args.perc_stories = 100
    args.num_bins = 4
    
    

    custom_data = [0,1,2,1,0,2,0,0,1,0,1,2,0,1,2,0,1,2,0,2,0]
    vocab_size = 3
    context_length = 5
    batch_size = 2
    stride = 1
    root_ctx_len = 2
    

    # Example usage
    dataset_dir = '/scratch/st-cthrampo-1/vaalaa/NTP_LLM_TokenDist/WikiText'  # Your saved dataset folder
    print("Running experiments for Vocab Size " + str(vocab_size) + " with Context Lenght " + str(context_length))

    args.context_length = context_length
    args.vocab_size = vocab_size
    args.batch_size = batch_size
    args.stride = stride
    args.root_ctx_len = root_ctx_len


    # Example usage
    # custom_data = [1,2,3,2,1,3,1,1,2,1,2,3,1,2,3,1,2,3,1,3,1]
    # custom_data = [1,2,1,3,1,3,2,2]
    # vocabs = [i for i in range(0,vocab_size)]
    # prob_dist = heavy_tail_prob_dist(vocab_size, 0.1)
    # sequence_range = 10000
    # custom_data = [np.random.choice(vocabs, p=prob_dist) for _ in range(sequence_range)]  # Replace 10 with your desired length
    
    
    
    dataloader, num_ctx = create_integer_dataloader(custom_data, context_length, batch_size, stride)
    
    # Create a bin of size 4
    firstTwo_token_bins = {}
    # Training loop for the first model on dataset1
    for batch in tqdm(dataloader):  
        x_batch_firstTwoTokens = batch[:, 0:2] 
        x_batch_firstTwoTokens = [tuple(row) for row in x_batch_firstTwoTokens.tolist()]
        firstTwo_token_bins = update_firstTwo_count_dict(x_batch_firstTwoTokens, firstTwo_token_bins)

    save_graph_folder = "/scratch/st-cthrampo-1/vaalaa/NTP_LLM_DataStats_Trie_MultiProcessor/Debug_Files/Graphs/firstTokenDistributions/"
    filename = f"voc{args.vocab_size}_ctxLen{args.context_length}_{args.perc_stories}%TS_Stride{args.stride}"
    graph_tokenFirstTwoDist_filename = f"{save_graph_folder}{filename}_firstTwoTokens.jpg"
    plot_tuple_frequency(firstTwo_token_bins, graph_tokenFirstTwoDist_filename)

    bins, bin_sums = distribute_tuples(firstTwo_token_bins, args.num_bins)
    print("Bin loads:", bin_sums)
    # print("Bin assignments:", bins)


    save_Trie_folder = "/scratch/st-cthrampo-1/vaalaa/NTP_LLM_DataStats_Trie_MultiProcessor/Debug_Files/Tries/"
    folder_name_Tries = filename + f"_NumBins{args.num_bins}/"
    folder_Tries_path = save_Trie_folder + folder_name_Tries
    if not os.path.exists(folder_Tries_path):
        os.mkdir(folder_Tries_path)
    
    for b in range(0, args.num_bins):
        bin_folder_path = folder_Tries_path + f"group{b}/"
        if not os.path.exists(bin_folder_path):
            os.mkdir(bin_folder_path)
        if not os.path.exists(bin_folder_path + "context_trees_memap_cpp/"):
            os.mkdir(bin_folder_path + "context_trees_memap_cpp/")
        if not os.path.exists(bin_folder_path + "graph_trees_cpp/"):
            os.mkdir(bin_folder_path + "graph_trees_cpp/")
        if not os.path.exists(bin_folder_path + "logs_trees_cpp/"):
            os.mkdir(bin_folder_path + "logs_trees_cpp/")

        bin_assigned_indices = bins[b]
        np.save(bin_folder_path + 'indices.npy', bin_assigned_indices)
    

    bin_folder_path = folder_Tries_path + f"group_root/"
    if not os.path.exists(bin_folder_path):
        os.mkdir(bin_folder_path)
    if not os.path.exists(bin_folder_path + "context_trees_memap_cpp/"):
        os.mkdir(bin_folder_path + "context_trees_memap_cpp/")
    if not os.path.exists(bin_folder_path + "graph_trees_cpp/"):
        os.mkdir(bin_folder_path + "graph_trees_cpp/")
    if not os.path.exists(bin_folder_path + "logs_trees_cpp/"):
        os.mkdir(bin_folder_path + "logs_trees_cpp/")

    

    entropy_branches_Trie = {}
    entropy_branches_Dict = {}
    for group in range(0, args.num_bins):
        print("_" * 100)
        bin_assigned_indices =  bins[group]
        print("bins[group]: " + str(bins[group]))
        dataloader, num_ctx = create_shard_integer_dataloader(custom_data, context_length, batch_size, stride, bin_assigned_indices, args.root_ctx_len, is_root = False)
        filename = f"voc{args.vocab_size}_ctxLen{args.context_length}_{args.perc_stories}%TS_Stride{args.stride}"
        save_Trie_folder = "/scratch/st-cthrampo-1/vaalaa/NTP_LLM_DataStats_Trie_MultiProcessor/Debug_Files/Tries/"
        folder_name_Tries = filename + f"_NumBins{args.num_bins}/"
        folder_Tries_path = save_Trie_folder + folder_name_Tries
        bin_folder_path = folder_Tries_path + f"group{group}/"
        args.group = group
        num_milestones = 2
        print("num_ctx: " + str(num_ctx))
        context_tree = load_or_create_tree(args, bin_folder_path, dataloader, num_milestones, num_ctx, is_root = False)

        # print("Tree loading/contruction complete")
        # result = context_tree.calculate_and_get_entropy_faster_branch()
        # dataset_entropy = result.entropy
        # total_count = result.total_count
        # print("Entropy Calculated Trie: " + str(dataset_entropy))
        # print("Count Calculated Trie: " + str(total_count))
        # print("_" * 100)

        # entropy_branches_Trie[f"Group_{group}"]={"entropy": dataset_entropy, "count": total_count}
        

        # entropy, total_num = cal_entropy_on_python_dict(dataloader, args.context_length - args.root_ctx_len, is_root = False)
        # print("Entropy Calculated Dict: " + str(entropy))
        # print("Count Calculated Dict: " + str(total_num))
        # entropy_branches_Dict[f"Group_{group}"]={"entropy": entropy, "count": total_num}
        # print("_" * 100)
        # # context_tree.cleanup()
        # # context_tree = None
       
        # pdb.set_trace()
        context_tree.reset()  # Explicitly clean up
        context_tree = None 
        # del context_tree
        input()

    
    print("_" * 100)
    print("Root")
    dataloader, num_ctx = create_shard_integer_dataloader(custom_data, context_length, batch_size, stride, None, args.root_ctx_len, is_root = True)
    filename = f"voc{args.vocab_size}_ctxLen{args.context_length}_{args.perc_stories}%TS_Stride{args.stride}"
    save_Trie_folder = "/scratch/st-cthrampo-1/vaalaa/NTP_LLM_DataStats_Trie_MultiProcessor/Debug_Files/Tries/"
    folder_name_Tries = filename + f"_NumBins{args.num_bins}/"
    folder_Tries_path = save_Trie_folder + folder_name_Tries
    bin_folder_path = folder_Tries_path + f"group_root/"
    # args.group = group
    num_milestones = 2
    print("num_ctx: " + str(num_ctx))
    context_tree = load_or_create_tree(args, bin_folder_path, dataloader, num_milestones, num_ctx, is_root = True)

    print("Tree loading/contruction complete")
    result = context_tree.calculate_and_get_entropy_faster_root()
    dataset_entropy = result.entropy
    total_count = result.total_count
    print("Entropy Calculated Trie: " + str(dataset_entropy))
    print("Count Calculated Trie: " + str(total_count))
    print("_" * 100)

    entropy_branches_Trie[f"Root"]={"entropy": dataset_entropy, "count": total_count}
    
    
    entropy, total_num = cal_entropy_on_python_dict(dataloader, args.root_ctx_len, is_root = True)
    print("Entropy Calculated Dict: " + str(entropy))
    print("Count Calculated Dict: " + str(total_num))
    entropy_branches_Dict[f"Group_{group}"]={"entropy": entropy, "count": total_num}
    print("_" * 100)
    # input()


    full_entropy = 0
    full_count = 0
    for group, info in entropy_branches_Trie.items():
        full_entropy += info["entropy"] * info["count"]
        full_count += info["count"]
    full_entropy = full_entropy / full_count

    print(info)
    print("Full Entropy from multiProcessed Trie:" + str(full_entropy))
    print("Full Count from multiProcessed Trie:" + str(full_count))



    dataloader, num_ctx = create_integer_dataloader(custom_data, context_length, batch_size, stride)
    dict_counts = {}
    for batch in dataloader:
        for x in batch:
            for i in range(1,context_length+1):
                string = '-'.join(x[0:i].numpy().astype(str).tolist())
                if string in dict_counts.keys():
                    dict_counts[string][str(int(x[i].item()))] += 1
                else:
                    dict_counts[string] = {str(i):0 for i in range(0,vocab_size)}
                    dict_counts[string][str(int(x[i].item()))] += 1

    entropy = 0
    total_num = 0
    entropy_dict = {}
    for index, (ctx, ctx_ctp) in enumerate(dict_counts.items()):
        entropy_ctx = 0
        count_ctx = sum(list(ctx_ctp.values()))
        for _, (ntp, count) in enumerate(ctx_ctp.items()):
            if count != 0:
                entropy_ctx -= count / count_ctx * np.log(count / count_ctx)
        entropy += count_ctx * entropy_ctx
        total_num += count_ctx
        
        entropy_dict[ctx] = entropy_ctx
    
    entropy /= total_num
    entropy_dict = {key:value / total_num for key,value in entropy_dict.items()}
    
    # print("dict_counts:")
    # print(dict_counts)
    print("entropy String dataset: " + str(entropy))
    print("num of contexts String dataset: " + str(len(list(dict_counts.keys()))))
    # print("entropy_dict: " + str(entropy_dict))
    # print("num unique contexts: " + str(len(dict_counts.keys())))
    
    # input()



    
    # print("Analyzing Data ... ")
    # # Manual training loop
    # num_datapoints = 0

    
    # count_num_one_hots = 0
    
    # norm_soft_vs_hard_diff = 0
    # num_total_samples = 0

    # print("_" * 100)
    # print("Checking that the Trie is correct")
    # is_correct = True
    # # Training loop for the first model on dataset1
    # for batch in tqdm(dataloader):

    #     y_one_hot = F.one_hot(batch[:, 1:], num_classes=args.vocab_size).float()  # Assuming vocab_size is defined
    #     y_soft_label = get_soft_label(context_tree, args, batch).float()

    #     x_batch = batch[:, :-1]  # Everything except the last token
    #     norms = torch.norm(y_one_hot - y_soft_label, p=1, dim=-1)
    #     norm_soft_vs_hard_diff += norms.sum()
    #     num_total_samples += batch.shape[0] * batch.shape[1]

    #     for i in range(0, batch.shape[0]):
    #         for context_index in range(1,batch.shape[1]):
    #             string = '-'.join(batch[i,0:context_index].numpy().astype(str).tolist())
                
    #             prob_vector_trie = y_soft_label[i,context_index-1]
    #             # Step 1: Extract values as a tensor
    #             counts = torch.tensor([dict_counts[string].get(str(i), 0) for i in range(vocab_size)], dtype=torch.float32)
    #             # Step 2: Normalize to create a probability vector
    #             prob_vector = counts / counts.sum()

    #             # print(y_soft_label[i,context_index-1])
    #             # print(prob_vector)
    #             if not torch.allclose(prob_vector_trie, prob_vector, atol=1e-6):
    #                 print("Following context are incorrrectly calculated with the Trie.")
    #                 print("Context: " + str(string))
    #                 print("soft_label tree: " + str(y_soft_label[i,context_index-1]))
    #                 print("soft_label python: " + str(prob_vector))
    #                 print("_______________________________________________________________________")
    #                 is_correct = False
    
    # if not is_correct:
    #     print("_" * 100)
    #     print("Trie is contructed incorrectly")
    #     print("_" * 100)

    # print("precentage of one hots: " + str(round(count_num_one_hots / num_total_samples * 100, 3)))


    # print("_" * 100)
    # print("Deleting tree and reloading it ...")
    # del context_tree
    # print("Deleted")
    # print("_" * 100)

    # memap_filename = f"debug_trie"
    # save_tree_folder =  "/scratch/st-cthrampo-1/vaalaa/NTP_LLM_TrieMultithreaded_TS_debugFiles/TrieFiles/context_trees_memap_cpp/"
    # memap_filename = f"{save_tree_folder}{memap_filename}"
    # # input(memap_filename)
    # context_tree = trie_module_protV1_lib_multithreaded.Trie_module_protV1(memap_filename)
    # print("Reloaded")
    # print("_" * 100)


    # is_correct = True
    # print("_" * 100)
    # print("Checking that the loaded Trie is correct")
    # # Training loop for the first model on dataset1
    # for batch in tqdm(dataloader):

    #     y_one_hot = F.one_hot(batch[:, 1:], num_classes=args.vocab_size).float()  # Assuming vocab_size is defined
    #     y_soft_label = get_soft_label(context_tree, args, batch).float()

    #     x_batch = batch[:, :-1]  # Everything except the last token
    #     norms = torch.norm(y_one_hot - y_soft_label, p=1, dim=-1)
    #     norm_soft_vs_hard_diff += norms.sum()
    #     num_total_samples += batch.shape[0] * batch.shape[1]

    #     for i in range(0, batch.shape[0]):
    #         for context_index in range(1,batch.shape[1]):
    #             string = '-'.join(batch[i,0:context_index].numpy().astype(str).tolist())
                
    #             prob_vector_trie = y_soft_label[i,context_index-1]
    #             # Step 1: Extract values as a tensor
    #             counts = torch.tensor([dict_counts[string].get(str(i), 0) for i in range(vocab_size)], dtype=torch.float32)
    #             # Step 2: Normalize to create a probability vector
    #             prob_vector = counts / counts.sum()

    #             # print(y_soft_label[i,context_index-1])
    #             # print(prob_vector)
    #             if not torch.allclose(prob_vector_trie, prob_vector, atol=1e-6):
    #                 print("Following context are incorrrectly calculated with the Trie.")
    #                 print("Context: " + str(string))
    #                 print("soft_label tree: " + str(y_soft_label[i,context_index-1]))
    #                 print("soft_label python: " + str(prob_vector))
    #                 print("_______________________________________________________________________")
    #                 input()

    # if not is_correct:
    #     print("_" * 100)
    #     print("Trie is reloaded incorrectly")
    #     print("_" * 100)

    # print("Experiment complete!")





# /scratch/st-cthrampo-1/vaalaa/NTP_LLM_TrieMultithreaded_TS_debugFiles/TrieFiles/context_trees_memap_cpp/debug_trie

# /scratch/st-cthrampo-1/vaalaa/NTP_LLM_TrieMultithreaded_TS_debugFiles/TrieFiles/context_trees_memap_cpp/debug_trie