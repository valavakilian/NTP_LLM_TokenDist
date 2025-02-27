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
import trie_module_protV1_lib
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


# from TS_loader import *

from TS_loader_memap import *
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


    save_tree_folder =  "/scratch/st-cthrampo-1/vaalaa/NTP_LLM_TinyStories/Trie_CPP_saves/context_trees_memap_cpp/"
    save_graph_folder = "/scratch/st-cthrampo-1/vaalaa/NTP_LLM_TinyStories/Trie_CPP_saves/graph_trees_cpp/"
    save_logs_folder =  "/scratch/st-cthrampo-1/vaalaa/NTP_LLM_TinyStories/Trie_CPP_saves/logs_trees_cpp/"
    save_logs_filename = f"voc_{args.vocab_size}_ctxLen_{args.context_length}.pkl"
    memap_filename = f"{save_tree_folder}{memap_filename}"

       
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

    try:
        
        # if exp_was_initiated: 
        #     print("Tree was prev initiated but incomplete. Loading ...")
        #     up_to_ctx_count_pbatchrocessed = list(data_log["entropy"].keys())[-1]
        #     context_tree = trie_module_protV1_lib.Trie_module_protV1(memap_filename)

        #     entropy_tree = context_tree.calculate_and_get_entropy()
        #     print("Entropy so far is is : " + str(entropy_tree))
        # else:
        print("Tree is new. Constructing ...")
        up_to_ctx_count_processed = 0
        context_tree = trie_module_protV1_lib.Trie_module_protV1(memap_filename, 200, args.context_length)

        data_log = {
            "entropy": {},
            "softlabel_diff": {},
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
                start_time_insert = time.time()
                _ = context_tree.insert(X, False)
                insert_runtime += time.time() - start_time_insert
                del X
                # print("Inserted a batch")
                

                if milestone_index < len(milestones) and contexts_count >= milestones[milestone_index]:
                    
                    num_ctx_seen = milestones[milestone_index]

                    data_log["insert_calc_time"][contexts_count] = insert_runtime
                    
                    print(f"Current Trie memory usage: {context_tree.get_memory_usage()//(1024)**3} GB")
                    
                    print(f"Inserting on old trie took: {data_log['insert_calc_time'][contexts_count]} seconds.")
                    print("_"*30)
 
                    print("_"*30)
                    start_time_entropy = time.time()
                    entropy_tree_new, softlabel_diff = context_tree.calculate_and_get_entropy_faster()
                    data_log["entropy_calc_time"][contexts_count] = time.time() - start_time_entropy
                    print("Entropy faster: " + str(entropy_tree_new))
                    print("softlabel_diff faster: " + str(softlabel_diff))
                    print("Took " + str(time.time() - start_time_entropy) + " sec.")
                    data_log["entropy"][contexts_count] = entropy_tree_new
                    data_log["softlabel_diff"][contexts_count] = softlabel_diff
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


                    plot_data_log_subplots(data_log, save_graph_folder + f"logs_voc_{args.vocab_size}_ctxLen_{args.context_length}_stride{args.stride}_{args.perc_stories}%TS.jpg", precDone = round(batches_seen / len(dataloader) * 100, 2))
                    plot_calc_times(data_log, save_graph_folder + f"runtime_voc_{args.vocab_size}_ctxLen_{args.context_length}_stride{args.stride}_{args.perc_stories}%TS.jpg", precDone = round(batches_seen / len(dataloader) * 100, 2))
                    plot_entropy_perCtxLen(data_log, save_graph_folder + f"entropy_voc_{args.vocab_size}_ctxLen_{args.context_length}_stride{args.stride}_{args.perc_stories}%TS.jpg", precDone = round(batches_seen / len(dataloader) * 100), ctx_len = args.context_length)

                    with open(save_logs_folder + save_logs_filename, 'wb') as pickle_file:
                        pickle.dump(data_log, pickle_file)
                    

                    milestone_index += 1
                    insert_runtime = 0
            
                    context_tree.save_metadata()
    
    except RuntimeError as e:
        print(f"An error occurred: {e}")
    
    return context_tree



def get_soft_label(context_tree, args, X):
    # print(X)
    output = context_tree.retrieve_softlabel(X)

    # print(output)
    y_soft_label = torch.zeros(X.shape[0], args.context_length, args.vocab_size)
    for data_index, soft_label_list in enumerate(output):
        # For each key in the dictionary, set the value in the tensor to 1
        for ctx_index, ctx_dict in enumerate(soft_label_list[0:-1]):
            for vocab_index, vocab_count in ctx_dict.items():
                y_soft_label[data_index, ctx_index, vocab_index] = vocab_count
    y_soft_label = F.normalize(y_soft_label, p = 2, dim = 2)

    return y_soft_label

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
    parser.add_argument("--stride", type=int, default=1, help="Step size")
    parser.add_argument("--perc_stories", type=int, default=100, help="percentage of stories")
    parser.add_argument("--scheduler_type", type=str, default="cosine", help="lr-scheduling style")
    args = parser.parse_args()

    

    # vocab_sizes = [64, 128, 256, 512, 1024, 2048, 4096, 8196, 16392]
    # context_lenghts = [32, 64, 128, 256, 512, 1024]

    # total_num_cases = len(vocab_sizes) * len(context_lenghts)
    # args.vocab_size = vocab_sizes[args.exp_case % len(vocab_sizes)]
    # args.context_length = context_lenghts[args.exp_case // len(vocab_sizes)]
    # args.stride = args.context_length // 2
    num_epochs = 90

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
        tokenizer_dir="/scratch/st-cthrampo-1/vaalaa/NTP_LLM_TinyStories/data_saves/tokenizers/tokenizer_output_dir",
        output_dir=f"/scratch/st-cthrampo-1/vaalaa/NTP_LLM_TinyStories/data_saves/tokenized_data"
    )

    print("Data processed. ")

    dataset = TinyStoriesDataset(
        data_dir=f"/scratch/st-cthrampo-1/vaalaa/NTP_LLM_TinyStories/data_saves/tokenized_data",
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


    num_milestones = 100
    memap_filename = f"Trie_voc{args.vocab_size}_ctxLen{args.context_length}_stride{args.stride}_{int(args.perc_stories)}%TS"
    context_tree = load_or_create_tree(args, memap_filename, dataloader, num_milestones, num_ctx)
    print("Tree loading/contruction complete")
    dataset_entropy = context_tree.calculate_and_get_entropy_faster()
    print("Entropy Calculated: " + str(dataset_entropy))

    
    # print("Loading the model to see if the issue is with the model")
    # # Model configuration (same model setup for both)
    # model_config = GPT2Config(
    #     vocab_size=args.vocab_size,
    #     n_positions=args.context_length,
    #     n_embd=512,
    #     n_layer=4,
    #     n_head=4,
    # )

    
    # # Initialize two separate models
    # model_one_hot = GPT2LMHeadModel(model_config)
    # model_soft_label = GPT2LMHeadModel(model_config)
    # print("Model created on cpu ...")
    
    # # Copy the weights from model_one_hot to model_soft_label
    # model_soft_label.load_state_dict(model_one_hot.state_dict())

    # model_one_hot = torch.compile(model_one_hot)
    # model_soft_label = torch.compile(model_soft_label)


    # # Move models to GPU if available
    # # device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


    # print("Analyzing Data ... ")
    # # Manual training loop
    # num_datapoints = 0
    

    # count_num_one_hots = 0
    # num_total_samples = 0
    # # Training loop for the first model on dataset1
    # for batch in dataloader:
        
    #     # Extract `x` (input) and `y` (target) from the full sequence
    #     x_batch = batch[:, :-1]  # Everything except the last token
    #     # print(batch[:, 1:].shape)
    #     # print(batch[:, 1:])
    #     y_one_hot = F.one_hot(batch[:, 1:], num_classes=args.vocab_size).float()  # Assuming vocab_size is defined
    #     y_soft_label = get_soft_label(context_tree, args, batch).float()

    #     # print(y_soft_label)
    #     # print(y_soft_label.shape)

    #     # Count non-zero entries along the last dimension
    #     non_zero_counts = (y_soft_label != 0).sum(dim=-1)

    #     # Check if only one non-zero per example and count them
    #     single_non_zero_count = (non_zero_counts == 1).sum()

    #     # input()
    #     # print(non_zero_counts)
    #     # print(non_zero_counts.shape)
    #     # input()
    #     # print(single_non_zero_count)
    #     # print(single_non_zero_count.shape)
    #     # input()
    #     count_num_one_hots += single_non_zero_count.item()
    #     num_total_samples += batch.shape[0] * batch.shape[1]

    print("precentage of one hots: " + str(round(count_num_one_hots / num_total_samples * 100, 3)))

    print("Training complete!")

        

    