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



# from Wiki_loader import *

from OpenWebText_loader_memap_sharded import *
# -----------------------------------------------------------------------------
# CLI for constructing the dataset

from timeit import default_timer as timer



print("Importing Done")


# -----------------------------------------------------------------------------
# CLI for constructing the dataset

SIZE_NODE_BYTES = 56 

def plot_calc_times(data_log, file_path, precDone):
    # Extract the values for each key across document counts (assuming document count is the key)
    doc_counts = sorted(data_log['insert_calc_time'].keys())  # Assuming the keys are document counts
    
    insert_calc_time = [data_log['insert_calc_time'][doc] for doc in doc_counts]
    
    # Create a plot
    plt.figure(figsize=(10, 6))

    # Plot insert_calc_time
    plt.plot(doc_counts, insert_calc_time, label='Insert Calculation Time Old', marker='s', color='orange')
    
    # Log scale on both axes
    # plt.xscale('log')
    plt.yscale('log')
    
    # Add labels and title
    plt.xlabel('Number of Documents (log scale)')
    plt.ylabel('Calculation Time (log scale)')
    plt.title('Insert Time over #Ctx Seen (' + str(precDone) + ' % ' + 'complete)')

    # Add text box for time differences
    textstr = f'Time Insert Calc: {insert_calc_time[-1]:.2f}'
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


def calculate_trie_size_gb(num_contexts: int, context_length: int, avg_children_per_node: int = 3) -> int:
    """
    Calculate trie size based on actual C++ structure sizes
    
    Args:
        num_contexts: Number of sequences being input
        context_length: Length of each context sequence
        avg_children_per_node: Average number of children per non-leaf node
    """
    # Constants from C++ code
    TRIE_NODE_SIZE = 32  # sizeof(TrieNode)
    CHILD_ENTRY_SIZE = 16  # sizeof(pair<int64_t, int64_t>)
    BYTES_TO_GB = 1024 ** 3
    
    print("\nMemory Calculation Breakdown:")
    print(f"Input Parameters:")
    print(f"- Number of contexts: {num_contexts:,}")
    print(f"- Context length: {context_length}")
    print(f"- Avg children per node: {avg_children_per_node}")
    
    # Calculate maximum nodes
    # Each context can create up to context_length nodes
    max_nodes = num_contexts * context_length
    node_storage = max_nodes * TRIE_NODE_SIZE
    
    print(f"\nNode Storage:")
    print(f"- Max possible nodes: {max_nodes:,}")
    print(f"- Storage per node: {TRIE_NODE_SIZE} bytes")
    print(f"- Total node storage: {node_storage:,} bytes ({node_storage/BYTES_TO_GB:.2f} GB)")
    
    # Calculate child entry storage
    # Each non-leaf node has space for child entries
    non_leaf_nodes = max_nodes - num_contexts  # leaf nodes don't need child entries
    child_storage = non_leaf_nodes * avg_children_per_node * CHILD_ENTRY_SIZE
    
    print(f"\nChild Entry Storage:")
    print(f"- Non-leaf nodes: {non_leaf_nodes:,}")
    print(f"- Storage per child entry: {CHILD_ENTRY_SIZE} bytes")
    print(f"- Total child storage: {child_storage:,} bytes ({child_storage/BYTES_TO_GB:.2f} GB)")
    
    # Additional overhead for memory management and alignment
    overhead_factor = 1.5  # 50% overhead for memory management
    total_storage = (node_storage + child_storage) * overhead_factor
    
    print(f"\nTotal Storage:")
    print(f"- Raw storage: {(node_storage + child_storage):,} bytes ({(node_storage + child_storage)/BYTES_TO_GB:.2f} GB)")
    print(f"- With {overhead_factor}x overhead: {total_storage:,} bytes ({total_storage/BYTES_TO_GB:.2f} GB)")
    
    # Round up to nearest GB and add safety margin
    size_gb = int(np.ceil(total_storage / BYTES_TO_GB)) + 10
    
    return size_gb



def load_or_create_tree(args, bin_folder_path, dataloader, num_milestones, num_examples):

    milestones = generate_equal_spaced_points(num_examples, num_milestones)[1:] # exclude zero
    print("milestones are : " + str(milestones))

    if not os.path.exists(bin_folder_path + "context_trees_memap_cpp/"):
        os.mkdir(bin_folder_path + "context_trees_memap_cpp/")
    if not os.path.exists(bin_folder_path + "graph_trees_cpp/"):
        os.mkdir(bin_folder_path + "graph_trees_cpp/")
    if not os.path.exists(bin_folder_path + "logs_trees_cpp/"):
        os.mkdir(bin_folder_path + "logs_trees_cpp/")

    save_tree_folder =  bin_folder_path + "context_trees_memap_cpp/"
    save_graph_folder = bin_folder_path + "graph_trees_cpp/"
    save_logs_folder =  bin_folder_path + "logs_trees_cpp/"
    save_logs_filename_MT = f"Trie{args.group}.pkl"
    memap_filename_MT = f"{save_tree_folder}Trie{args.group}_MT"


    # Trie_predicted_size = max(int(SIZE_NODE_BYTES * num_examples * (args.context_length + 1) * 5 // (1024**3)), 60)
    # Predict trie size
    num_examples = num_examples * args.batch_size
    print("dataloader lenghts: " + str(num_examples))
    Trie_predicted_size = calculate_trie_size_gb(num_examples, args.context_length)

       
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
        print(f"Trie is of size {Trie_predicted_size} GB")
        context_tree_MT = trie_module_protV1_lib_multithreaded.Trie_module_protV1(memap_filename_MT, Trie_predicted_size, args.context_length)



    data_log_MT = {
        "entropy": {},
        "total_count": {},
        "entropy_per_ctx_len": {},
        "num_total_ctx": {},
        "num_unique_ctx": {},
        "num_unique_ctx_len_list": {},
        "num_total_ctx_len_list": {},
        "insert_calc_time": {},
        "entropy_calc_time": {},
        "num_oneHots_list": {},
        "supSize_list": {},
        "uniformity_list": {}
    }


    insert_runtime_MT = 0
    insert_runtime_ST = 0
    
    contexts_count = 0
    milestone_index = 0
    # Step 6: Iterate through the DataLoader and print samples
    batches_seen = 0

    # print("HEHEHEHEHEHEHEHEH")
    
    for X in tqdm(dataloader):

        # print("HEHEHEHEHEHEHEHEH")
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

            if milestone_index < len(milestones) and batches_seen >= milestones[milestone_index]:
                
                num_ctx_seen = milestones[milestone_index]

                data_log_MT["insert_calc_time"][contexts_count] = insert_runtime_MT                
                print(f"Current MT Trie memory usage: {context_tree_MT.get_memory_usage()//(1024)**3} GB")
                
                print(f"Inserting MT trie took: {data_log_MT['insert_calc_time'][contexts_count]} seconds.")
                print("_"*30)
                
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
    parser.add_argument("--group", type=int, default=0, help="Used for sockeye purposes")
    parser.add_argument("--batch_size", type=int, default=512, help="Batch size")
    parser.add_argument("--stride", type=int, default=1, help="Window stride size")
    parser.add_argument("--perc_stories", type=int, default=100, help="percentage of stories")
    parser.add_argument("--scheduler_type", type=str, default="cosine", help="lr-scheduling style")
    parser.add_argument("--num_epochs", type=int, default=90, help="Step size")
    parser.add_argument("--LoadTrieFromFile", type=bool, default=False, help="Load from existing file")
    parser.add_argument("--num_bins", type=int, default=4, help="Number of shard")
    parser.add_argument("--root_ctx_len", type=int, default=2, help="Size of the root context lenght, shards will each have context length of (context_length - root_ctx_len) for the Trie")
    args = parser.parse_args()

    # with open('/scratch/st-cthrampo-1/vaalaa/NTP_LLM_DataStats_Trie_MultiProcessor_OpenWebText/outputs/mylogtext.txt', 'w') as file:
    #     file.write("Got in ? \n")

    dataset_dir = '/arc/project/st-cthrampo-1/vala/openwebtext_karpathy/nanoGPT/data/openwebtext/train.bin'  # Your saved dataset folder

    # Step 4: Load and Tokenize the Wikitext-2 Dataset
    # Example usage with stride
    print("_" * 100)
    print("Creating dataloader ... ")
    dataloader, vocab_size = create_dataloader(
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
    args.vocab_size = vocab_size

    print("Running experiments for Vocab Size " + str(args.vocab_size) + " with Context Lenght " + str(args.context_length))
    

    # Example usage
    vocab_size = args.vocab_size
    context_length = args.context_length
    batch_size = args.batch_size
    perc_stories = args.perc_stories

    filename = f"voc{args.vocab_size}_ctxLen{args.context_length}_{args.perc_stories}%OpWT_Stride{args.stride}"
    save_Trie_folder = "/scratch/st-cthrampo-1/vaalaa/NTP_LLM_DataStats_Trie_MultiProcessor_OpenWebText/Tries/"
    folder_name_Tries = filename + f"_NumBins{args.num_bins}/"
    folder_Tries_path = save_Trie_folder + folder_name_Tries
    bin_folder_path = folder_Tries_path + f"group{args.group}/"

    local_bin_folder_path = "./Trie_info/"
    if not os.path.exists(local_bin_folder_path):
        os.mkdir(local_bin_folder_path)
    print(f"Directory exists: {os.path.exists(local_bin_folder_path)}")
    bin_assigned_indices = np.load(bin_folder_path + 'indices.npy')


    # Step 4: Load and Tokenize the Wikitext-2 Dataset
    # Example usage with stride
    num_ctx = len(dataloader)

    num_milestones = 100    
    context_tree = load_or_create_tree(args, local_bin_folder_path, dataloader, num_milestones, num_ctx)
    print("Tree loading/contruction complete")

    save_trie_time = time.time()
    context_tree.serialize_to_mmap()
    save_trie_time = save_trie_time - time.time()
    print(f"Took {save_trie_time} time to save trie." )



        

    