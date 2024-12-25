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



from Wiki_loader_memap_sharded import *
# -----------------------------------------------------------------------------
# CLI for constructing the dataset

from timeit import default_timer as timer



print("Importing Done")


# -----------------------------------------------------------------------------
# CLI for constructing the dataset


def plot_oneHot_perCtxLen(data_log, file_path, precDone, ctx_len):
    # Extract the values for each key across document counts (assuming document count is the key)
    doc_counts = sorted(data_log['num_oneHots_list'].keys())  # Assuming the keys are document counts
    
    target_ctx_lens = np.linspace(1, len(ctx_len) // 2, 16, dtype=int).tolist()
    target_ctx_lens = [ctx_len[t] for t in target_ctx_lens]
    target_ctx_lens.append(3)
    target_ctx_lens.append(5)
    target_ctx_lens.append(8)
    target_ctx_lens = list(set(target_ctx_lens))
    target_ctx_lens.sort()
    # target_ctx_lens = [4,8,12,16,24,32,64]
    
    oneHot = [sum(data_log['num_oneHots_list'][doc][t] for t in range(1, args.context_length + 1)) / sum(data_log['num_total_ctx_len_list'][doc][t] for t in range(1, args.context_length + 1)) for doc in doc_counts]
    oneHot_per_ctxLen = {}
    for t in target_ctx_lens:
        oneHot_per_ctxLen[t] = [data_log['num_oneHots_list'][doc][t] / data_log['num_total_ctx_len_list'][doc][t] for doc in doc_counts]
    
    # Create subplots (3 rows, 1 column)
    fig, axs = plt.subplots(1, 1, figsize=(12, 10))
    
    # Plot oneHot
    axs.plot(doc_counts, oneHot, label='Total', marker='o', color = "black")
    for t in target_ctx_lens:
        axs.plot(doc_counts, oneHot_per_ctxLen[t], label=f't = {t}', marker='o')
    axs.set_xlabel('Number of Contexts Seen ')
    axs.set_ylabel('oneHot percentage')
    axs.set_title('oneHot over Contexts Seen(' + str(precDone) + ' % ' + 'complete)')
    axs.grid(True, which="both", ls="--")
    # Add text box for oneHot difference
    textstr = f'oneHot: {oneHot[-1]:.6f}'
    axs.text(0.05, 0.95, textstr, transform=axs.transAxes, fontsize=10, verticalalignment='top',
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    axs.legend(loc='best')
    # Adjust layout for better spacing
    plt.tight_layout()
    
    # Display the plot
    plt.savefig(file_path)
    plt.clf()





def plot_uniformity_perCtxLen(data_log, file_path, precDone, ctx_len):
    # Extract the values for each key across document counts (assuming document count is the key)
    doc_counts = sorted(data_log['entropy'].keys())  # Assuming the keys are document counts
    
    target_ctx_lens = np.linspace(1, len(ctx_len) // 2, 8, dtype=int).tolist()
    target_ctx_lens = [ctx_len[t] for t in target_ctx_lens]
    target_ctx_lens.append(3)
    target_ctx_lens.append(5)
    target_ctx_lens.append(8)
    target_ctx_lens = list(set(target_ctx_lens))
    target_ctx_lens.sort()
    
    # uniformity = [data_log['entropy'][doc] for doc in doc_counts]
    uniformity_per_ctxLen = {}
    for t in ctx_len:
        uniformity_per_ctxLen[t] = [data_log['uniformity_list'][doc][t] for doc in doc_counts]
    uniformity = [sum([uniformity_per_ctxLen[t][index] for t in ctx_len]) / len(ctx_len) for index in range(0, len(doc_counts))]
    
    # Create subplots (3 rows, 1 column)
    fig, axs = plt.subplots(1, 1, figsize=(12, 9))

    # Plot uniformity
    axs.plot(doc_counts, uniformity, label='Total', marker='o', color = "black")
    for t in target_ctx_lens:
        axs.plot(doc_counts, uniformity_per_ctxLen[t], label=f't = {t}', marker='o')
    axs.set_xscale('log')
    # axs.set_yscale('log')
    axs.set_xlabel('Number of Contexts Seen (log scale)')
    axs.set_ylabel('Uniformity (log scale)')
    axs.set_title('Uniformity over Contexts Seen(' + str(precDone) + ' % ' + 'complete)')
    axs.grid(True, which="both", ls="--")
    # Add text box for uniformity difference
    textstr = f'Uniformity: {uniformity[-1]:.6f}'
    axs.text(0.05, 0.95, textstr, transform=axs.transAxes, fontsize=10, verticalalignment='top',
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    axs.legend(loc='best')
    # Adjust layout for better spacing
    plt.tight_layout()
    
    # Display the plot
    plt.savefig(file_path)
    plt.clf()


def plot_supSize_perCtxLen(data_log, file_path, precDone, ctx_len):
    # Extract the values for each key across document counts (assuming document count is the key)
    doc_counts = sorted(data_log['entropy'].keys())  # Assuming the keys are document counts
    
    target_ctx_lens = np.linspace(1, len(ctx_len) // 2, 8, dtype=int).tolist()
    target_ctx_lens = [ctx_len[t] for t in target_ctx_lens]
    target_ctx_lens.append(3)
    target_ctx_lens.append(5)
    target_ctx_lens.append(8)
    target_ctx_lens = list(set(target_ctx_lens))
    target_ctx_lens.sort()
    
    # supSize = [data_log['entropy'][doc] for doc in doc_counts]
    supSize_per_ctxLen = {}
    for t in ctx_len:
        supSize_per_ctxLen[t] = [data_log['supSize_list'][doc][t] for doc in doc_counts]
    supSize = [sum([supSize_per_ctxLen[t][index] for t in ctx_len]) / len(ctx_len) for index in range(0, len(doc_counts))]
    
    # Create subplots (3 rows, 1 column)
    fig, axs = plt.subplots(1, 1, figsize=(12, 9))

    # Plot supSize
    axs.plot(doc_counts, supSize, label='Total', marker='o', color = "black")
    for t in target_ctx_lens:
        axs.plot(doc_counts, supSize_per_ctxLen[t], label=f't = {t}', marker='o')
    axs.set_xscale('log')
    # axs.set_yscale('log')
    axs.set_xlabel('Number of Contexts Seen (log scale)')
    axs.set_ylabel('Support Size (log scale)')
    axs.set_title('Support Size over Contexts Seen(' + str(precDone) + ' % ' + 'complete)')
    axs.grid(True, which="both", ls="--")
    # Add text box for supSize difference
    textstr = f'Support Size: {supSize[-1]:.6f}'
    axs.text(0.05, 0.95, textstr, transform=axs.transAxes, fontsize=10, verticalalignment='top',
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    axs.legend(loc='best')
    # Adjust layout for better spacing
    plt.tight_layout()
    
    # Display the plot
    plt.savefig(file_path)
    plt.clf()




def plot_entropy_perCtxLen(data_log, file_path, precDone, ctx_len):
    # Extract the values for each key across document counts (assuming document count is the key)
    doc_counts = sorted(data_log['entropy'].keys())  # Assuming the keys are document counts
    
    target_ctx_lens = np.linspace(1, len(ctx_len) // 2, 8, dtype=int).tolist()
    target_ctx_lens = [ctx_len[t] for t in target_ctx_lens]
    target_ctx_lens.append(3)
    target_ctx_lens.append(5)
    target_ctx_lens.append(8)
    target_ctx_lens = list(set(target_ctx_lens))
    target_ctx_lens.sort()
    
    entropy = [data_log['entropy'][doc] for doc in doc_counts]
    entropy_per_ctxLen = {}
    for t in target_ctx_lens:
        entropy_per_ctxLen[t] = [data_log['entropy_per_ctx_len'][doc][t] for doc in doc_counts]
    
    # Create subplots (3 rows, 1 column)
    fig, axs = plt.subplots(1, 1, figsize=(12, 9))

    # Plot entropy
    axs.plot(doc_counts, entropy, label='Total', marker='o', color = "black")
    for t in target_ctx_lens:
        axs.plot(doc_counts, entropy_per_ctxLen[t], label=f't = {t}', marker='o')
    axs.set_xscale('log')
    axs.set_yscale('log')
    axs.set_xlabel('Number of Contexts Seen (log scale)')
    axs.set_ylabel('Entropy (log scale)')
    axs.set_title('Entropy over Contexts Seen(' + str(precDone) + ' % ' + 'complete)')
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




# def plot_uniformity_perCtxLen(data_log, file_path, precDone, ctx_len):
#     # Extract the values for each key across document counts (assuming document count is the key)
#     doc_counts = sorted(data_log['entropy'].keys())  # Assuming the keys are document counts
    
#     target_ctx_lens = np.linspace(1, len(ctx_len) // 2, 8, dtype=int).tolist()
#     target_ctx_lens = [ctx_len[t] for t in target_ctx_lens]
#     target_ctx_lens.append(3)
#     target_ctx_lens.append(5)
#     target_ctx_lens.append(8)
#     target_ctx_lens = list(set(target_ctx_lens))
#     target_ctx_lens.sort()
    
#     entropy = [data_log['entropy'][doc] for doc in doc_counts]
#     entropy_per_ctxLen = {}
#     for t in target_ctx_lens:
#         entropy_per_ctxLen[t] = [data_log['entropy_per_ctx_len'][doc][t] for doc in doc_counts]
    
#     # Create subplots (3 rows, 1 column)
#     fig, axs = plt.subplots(1, 1, figsize=(12, 9))

#     # Plot entropy
#     axs.plot(doc_counts, entropy, label='Total', marker='o', color = "black")
#     for t in target_ctx_lens:
#         axs.plot(doc_counts, entropy_per_ctxLen[t], label=f't = {t}', marker='o')
#     axs.set_xscale('log')
#     axs.set_yscale('log')
#     axs.set_xlabel('Number of Contexts Seen (log scale)')
#     axs.set_ylabel('Entropy (log scale)')
#     axs.set_title('Entropy over Contexts Seen(' + str(precDone) + ' % ' + 'complete)')
#     axs.grid(True, which="both", ls="--")
#     # Add text box for entropy difference
#     textstr = f'Entropy: {entropy[-1]:.6f}'
#     axs.text(0.05, 0.95, textstr, transform=axs.transAxes, fontsize=10, verticalalignment='top',
#                 bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
#     axs.legend(loc='best')
#     # Adjust layout for better spacing
#     plt.tight_layout()
    
#     # Display the plot
#     plt.savefig(file_path)
#     plt.clf()




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
    axs[0].set_xlabel('Number of Contexts Seen (log scale)')
    axs[0].set_ylabel('Entropy (log scale)')
    axs[0].set_title('Entropy over Contexts Seen(' + str(precDone) + ' % ' + 'complete)')
    axs[0].grid(True, which="both", ls="--")
    # Add text box for entropy difference
    textstr = f'Entropy: {entropy[-1]:.6f}'
    axs[0].text(0.05, 0.95, textstr, transform=axs[0].transAxes, fontsize=10, verticalalignment='top',
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    # Plot num_total_ctx
    axs[1].plot(doc_counts, num_total_ctx, label='Total Contexts', marker='s', color='orange')
    axs[1].set_xscale('log')
    axs[1].set_yscale('log')
    axs[1].set_xlabel('Number of Contexts Seen (log scale)')
    axs[1].set_ylabel('Total Contexts (log scale)')
    axs[1].set_title('Total Contexts over Contexts Seen')
    axs[1].grid(True, which="both", ls="--")
    
    # Plot num_unique_ctx
    axs[2].plot(doc_counts, num_unique_ctx, label='Unique Contexts', marker='^', color='green')
    axs[2].set_xscale('log')
    axs[2].set_yscale('log')
    axs[2].set_xlabel('Number of Contexts Seen (log scale)')
    axs[2].set_ylabel('Unique Contexts (log scale)')
    axs[2].set_title('Unique Contexts over Contexts Seen')
    axs[2].grid(True, which="both", ls="--")
    
    # Adjust layout for better spacing
    plt.tight_layout()
    
    # Display the plot
    plt.savefig(file_path)
    plt.clf()






def plot_data_EntropyTotal(data_log, file_path, precDone):
    # Extract the values for each key across document counts (assuming document count is the key)
    doc_counts = sorted(data_log['entropy'].keys())  # Assuming the keys are document counts
    
    entropy = [data_log['entropy'][doc] for doc in doc_counts]
    num_total_ctx = [data_log['num_total_ctx'][doc] for doc in doc_counts]
    num_unique_ctx = [data_log['num_unique_ctx'][doc] for doc in doc_counts]
    
    # Create subplots (3 rows, 1 column)
    fig, axs = plt.subplots(1, 1, figsize=(12, 8))

    
    # Plot entropy
    axs.plot(doc_counts, entropy, label='Entropy Old', marker='o', color = "green")
    axs.set_xscale('log')
    axs.set_yscale('log')
    axs.set_xlabel('Number of Contexts Seen (log scale)')
    axs.set_ylabel('Entropy (log scale)')
    axs.set_title('Entropy over Contexts Seen(' + str(precDone) + ' % ' + 'complete)')
    axs.grid(True, which="both", ls="--")
    # Add text box for entropy difference
    textstr = f'Entropy: {entropy[-1]:.6f}'
    axs.text(0.05, 0.95, textstr, transform=axs.transAxes, fontsize=10, verticalalignment='top',
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    # Adjust layout for better spacing
    plt.tight_layout()
    
    # Display the plot
    plt.savefig(file_path)
    plt.clf()



def plot_data_UniqueTotal(data_log, file_path, precDone):
    # Extract the values for each key across document counts (assuming document count is the key)
    doc_counts = sorted(data_log['entropy'].keys())  # Assuming the keys are document counts
    
    entropy = [data_log['entropy'][doc] for doc in doc_counts]
    num_total_ctx = [data_log['num_total_ctx'][doc] for doc in doc_counts]
    num_unique_ctx = [data_log['num_unique_ctx'][doc] for doc in doc_counts]
    
    # Create subplots (3 rows, 1 column)
    fig, axs = plt.subplots(1, 1, figsize=(12, 8))

    
    # Plot entropy
    axs.plot(doc_counts, num_total_ctx, label='Entropy Old', marker='o', color = "green")
    axs.set_xscale('log')
    axs.set_yscale('log')
    axs.set_xlabel('Number of Contexts Seen (log scale)')
    axs.set_ylabel('Num Unique (log scale)')
    axs.set_title('Num Unique over Contexts Seen(' + str(precDone) + ' % ' + 'complete)')
    axs.grid(True, which="both", ls="--")
    # Add text box for entropy difference
    textstr = f'Num Unique: {num_unique_ctx[-1]:.6f}'
    axs.text(0.05, 0.95, textstr, transform=axs.transAxes, fontsize=10, verticalalignment='top',
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
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
    plt.xlabel('Number of Contexts Seen (log scale)')
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
    parser.add_argument("--num_bins", type=int, default=4, help="Step size")
    parser.add_argument("--root_ctx_len", type=int, default=2, help="Size of the root context lenght, shards will each have context length of (context_length - root_ctx_len) for the Trie")
    args = parser.parse_args()


    # with open('/scratch/st-cthrampo-1/vaalaa/NTP_LLM_TokenDist_WikiBig_multithreaded/outputs/mylogtext.txt', 'w') as file:
    #     file.write("Got in ? \n")


    full_trie_logs_dict = {}
    print("Running experiments for Vocab Size " + str(args.vocab_size) + " with Context Lenght " + str(args.context_length))
    
    filename = f"voc{args.vocab_size}_ctxLen{args.context_length}_{args.perc_stories}%Wiki_Stride{args.stride}"

    save_Trie_folder = "/scratch/st-cthrampo-1/vaalaa/NTP_LLM_DataStats_Trie_MultiProcessor_Wiki/Tries/"
    folder_name_Tries = filename + f"_NumBins{args.num_bins}/"
    folder_Tries_path = save_Trie_folder + folder_name_Tries
    
    # Load the shards
    for b in range(0, args.num_bins):
        bin_folder_path = folder_Tries_path + f"group{b}/"

        save_tree_folder =  bin_folder_path + "context_trees_memap_cpp/"
        save_graph_folder = bin_folder_path + "graph_trees_cpp/"
        save_logs_folder =  bin_folder_path + "logs_trees_cpp/"
        save_logs_filename_MT = f"Trie{b}.pkl"
        memap_filename_MT = f"{save_tree_folder}Trie{b}_MT"

        with open(save_logs_folder + save_logs_filename_MT, 'rb') as pickle_file:
            data_log_MT = pickle.load(pickle_file)
        
        full_trie_logs_dict["group"+str(b)] = data_log_MT
    
    # Load the roots
    bin_folder_path = folder_Tries_path + f"group_root/"

    save_tree_folder =  bin_folder_path + "context_trees_memap_cpp/"
    save_logs_folder =  bin_folder_path + "logs_trees_cpp/"
    save_logs_filename_MT = f"TrieRoot.pkl"
    memap_filename_MT = f"{save_tree_folder}TrieRoot_MT"

    with open(save_logs_folder + save_logs_filename_MT, 'rb') as pickle_file:
        data_log_MT = pickle.load(pickle_file)
    
    full_trie_logs_dict["groupRoot"] = data_log_MT


    # # Looking into the dataset
    # # Step 4: Load and Tokenize the Wikitext-2 Dataset
    # # Example usage with stride
    # print("_" * 100)
    # print("Training tokenizer and tokenizing data ... ")
    # tokenized_data, tokenizer = load_and_tokenize_wikitext(
    #     dataset_dir=dataset_dir,
    #     vocab_size=args.vocab_size,
    #     context_length=args.context_length,
    #     tokenizer_path="/scratch/st-cthrampo-1/vaalaa/NTP_LLM_DataStats_Trie_MultiProcessor_Wiki/Data/Wiki_tokenizer/",
    #     tokenized_data_path="/scratch/st-cthrampo-1/vaalaa/NTP_LLM_DataStats_Trie_MultiProcessor_Wiki/Data/Wiki_tokenized_dataset/"
    # )
    # print("Complete!")
    # print("_" * 100)
    




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
        "num_oneHots_len_list": {},
        "num_oneHots_list": {},
        "supSize_list": {},
        "uniformity_list": {}
    }
    
    for i in range(0, len(list(full_trie_logs_dict["groupRoot"]["entropy"].keys()))):
        entropy_iteration = 0
        count_iteration = 0
        num_total_ctx_iteration = 0
        num_unique_ctx_iteration = 0
        total_total_count = 0

        entropy_root = list(full_trie_logs_dict["groupRoot"]["entropy"].values())[i]
        count_root = list(full_trie_logs_dict["groupRoot"]["num_total_ctx"].values())[i]
        total_count = list(full_trie_logs_dict["groupRoot"]["total_count"].values())[i]
        entropy_iteration += entropy_root * total_count
        count_iteration += total_count
        num_total_ctx_iteration += list(full_trie_logs_dict["groupRoot"]["num_total_ctx"].values())[i]
        num_unique_ctx_iteration += list(full_trie_logs_dict["groupRoot"]["num_unique_ctx"].values())[i]
        
        entropy_per_ctx_len_dict = {}
        num_unique_ctx_len_list_dict = {}
        num_total_ctx_len_list_dict = {}
        num_oneHots_list_dict = {}
        supSize_list_dict = {}
        uniformity_list_dict = {}
        for ctx_len in range(1, args.context_length + 1):
            entropy_per_ctx_len_dict[ctx_len] = 0
            num_unique_ctx_len_list_dict[ctx_len] = 0
            num_total_ctx_len_list_dict[ctx_len] = 0
            num_oneHots_list_dict[ctx_len] = 0
            supSize_list_dict[ctx_len] = 0
            uniformity_list_dict[ctx_len] = 0
        
        for ctx_len in range(1, args.root_ctx_len + 1):
            # print(full_trie_logs_dict["groupRoot"]["entropy_per_ctx_len"])
            # print(list(full_trie_logs_dict["groupRoot"]["entropy_per_ctx_len"].keys()))
            # print(list(full_trie_logs_dict["groupRoot"]["entropy_per_ctx_len"].values())[i][ctx_len])
            # input()
            num_unique_ctx_len_list_dict[ctx_len] = list(full_trie_logs_dict["groupRoot"]["num_unique_ctx_len_list"].values())[i][ctx_len]
            num_total_ctx_len_list_dict[ctx_len] = list(full_trie_logs_dict["groupRoot"]["num_total_ctx_len_list"].values())[i][ctx_len]

            # if ctx_len == args.root_ctx_len:
            #     entropy_per_ctx_len_dict[ctx_len] = list(full_trie_logs_dict["groupRoot"]["entropy_per_ctx_len"].values())[i][ctx_len] / num_total_ctx_len_list_dict[ctx_len]
            # else:
            entropy_per_ctx_len_dict[ctx_len] = list(full_trie_logs_dict["groupRoot"]["entropy_per_ctx_len"].values())[i][ctx_len] 

            supSize_list_dict[ctx_len] = list(full_trie_logs_dict["groupRoot"]["supSize_list"].values())[i][ctx_len]
            uniformity_list_dict[ctx_len] = list(full_trie_logs_dict["groupRoot"]["uniformity_list"].values())[i][ctx_len]

            num_oneHots_list_dict[ctx_len] = list(full_trie_logs_dict["groupRoot"]["num_oneHots_list"].values())[i][ctx_len]
            

        # input("HERE")


        if i == 98:
            print("_" * 100)
            print("entropy_root: " + str(entropy_root))
            print("count_root: " + str(count_root))

        sum_bins = 0
        # for b in range(0, args.num_bins):
        for b in range(0, args.num_bins):

            # print(len(list(full_trie_logs_dict["group"+str(b)]["entropy"].values())))
            entropy_bin = list(full_trie_logs_dict["group"+str(b)]["entropy"].values())[i]
            count_bin = list(full_trie_logs_dict["group"+str(b)]["num_total_ctx"].values())[i]
            total_count = list(full_trie_logs_dict["group"+str(b)]["total_count"].values())[i]
            entropy_iteration += entropy_bin * total_count
            count_iteration += total_count

            num_total_ctx_iteration += list(full_trie_logs_dict["group"+str(b)]["num_total_ctx"].values())[i]
            num_unique_ctx_iteration += list(full_trie_logs_dict["group"+str(b)]["num_unique_ctx"].values())[i]


            for ctx_len in range(args.root_ctx_len + 1, args.context_length + 1):
                # print(list(full_trie_logs_dict["group"+str(b)]["entropy_per_ctx_len"].values())[i][ctx_len])
                # input()
                num_total_ctx_len_list_dict[ctx_len] += list(full_trie_logs_dict["group"+str(b)]["num_total_ctx_len_list"].values())[i][ctx_len]
                num_unique_ctx_len_list_dict[ctx_len] += list(full_trie_logs_dict["group"+str(b)]["num_unique_ctx_len_list"].values())[i][ctx_len]

                # if ctx_len == args.context_length:
                #     entropy_per_ctx_len_dict[ctx_len] += list(full_trie_logs_dict["group"+str(b)]["entropy_per_ctx_len"].values())[i][ctx_len] 
                # else:
                entropy_per_ctx_len_dict[ctx_len] += list(full_trie_logs_dict["group"+str(b)]["entropy_per_ctx_len"].values())[i][ctx_len] * num_total_ctx_len_list_dict[ctx_len]


                supSize_list_dict[ctx_len] += list(full_trie_logs_dict["group"+str(b)]["supSize_list"].values())[i][ctx_len] * num_total_ctx_len_list_dict[ctx_len]
                uniformity_list_dict[ctx_len] += list(full_trie_logs_dict["group"+str(b)]["uniformity_list"].values())[i][ctx_len] * num_total_ctx_len_list_dict[ctx_len]

                
                num_oneHots_list_dict[ctx_len] += list(full_trie_logs_dict["group"+str(b)]["num_oneHots_list"].values())[i][ctx_len]
                

            sum_bins += count_bin
            if i == 98:
                print("_" * 100)
                print(f"entropy_bin {b}: " + str(entropy_bin))
                print(f"count_bin {b}: " + str(count_bin))
        if i == 98:
            print("sum_bins: " + str(sum_bins))
        
        entropy_iteration /= count_iteration
        for ctx_len in range(args.root_ctx_len + 1, args.context_length + 1):
            entropy_per_ctx_len_dict[ctx_len] /= num_total_ctx_len_list_dict[ctx_len] 
            supSize_list_dict[ctx_len] /= num_total_ctx_len_list_dict[ctx_len] 
            uniformity_list_dict[ctx_len] /= num_total_ctx_len_list_dict[ctx_len] 
        

        data_log_MT["entropy"][count_iteration] = entropy_iteration
        data_log_MT["total_count"][count_iteration] = count_iteration
        data_log_MT["num_total_ctx"][count_iteration] = num_total_ctx_iteration
        data_log_MT["num_unique_ctx"][count_iteration] = num_unique_ctx_iteration

        data_log_MT["entropy_per_ctx_len"][count_iteration] = entropy_per_ctx_len_dict
        data_log_MT["supSize_list"][count_iteration] = supSize_list_dict
        data_log_MT["uniformity_list"][count_iteration] = uniformity_list_dict
        data_log_MT["num_unique_ctx_len_list"][count_iteration] = num_unique_ctx_len_list_dict
        data_log_MT["num_total_ctx_len_list"][count_iteration] = num_total_ctx_len_list_dict

        data_log_MT["num_oneHots_list"][count_iteration] = num_oneHots_list_dict
        

    print(list(data_log_MT["entropy"].values())[-1])
    print(list(data_log_MT["total_count"].values())[-1])
    print(sum(list(data_log_MT["entropy_per_ctx_len"].values())[-1].values()) /args.context_length)
    print(sum(list(data_log_MT["num_total_ctx_len_list"].values())[-1].values()))
    print(list(data_log_MT["entropy_per_ctx_len"].values())[-1])
    print(list(data_log_MT["num_total_ctx_len_list"].values())[-1])
    # input()

    save_full_graph_folder = folder_Tries_path + "full_graphs/"
    if not os.path.exists(save_full_graph_folder):
        os.mkdir(save_full_graph_folder)
    plot_data_log_subplots(data_log_MT, save_full_graph_folder + f"logs_voc_{args.vocab_size}_ctxLen_{args.context_length}_stride{args.stride}_{args.perc_stories}%Wiki.jpg", precDone = 100)
    plot_entropy_perCtxLen(data_log_MT, save_full_graph_folder + f"entropy_voc_{args.vocab_size}_ctxLen_{args.context_length}_stride{args.stride}_{args.perc_stories}%Wiki.jpg", precDone = 100, ctx_len = [t for t in range(1, args.context_length + 1)])
    plot_oneHot_perCtxLen(data_log_MT, save_full_graph_folder + f"OneHots_voc_{args.vocab_size}_ctxLen_{args.context_length}_stride{args.stride}_{args.perc_stories}%Wiki.jpg", precDone = 100, ctx_len = [t for t in range(1, args.context_length + 1)])
    plot_data_EntropyTotal(data_log_MT, save_full_graph_folder + f"EntropyTotal_voc_{args.vocab_size}_ctxLen_{args.context_length}_stride{args.stride}_{args.perc_stories}%Wiki.jpg", precDone = 100)
    plot_data_UniqueTotal(data_log_MT, save_full_graph_folder + f"NumUniqueTotal_voc_{args.vocab_size}_ctxLen_{args.context_length}_stride{args.stride}_{args.perc_stories}%Wiki.jpg", precDone = 100)
    plot_supSize_perCtxLen(data_log_MT, save_full_graph_folder + f"supSize_voc_{args.vocab_size}_ctxLen_{args.context_length}_stride{args.stride}_{args.perc_stories}%Wiki.jpg", precDone = 100, ctx_len = [t for t in range(1, args.context_length + 1)])
    plot_uniformity_perCtxLen(data_log_MT, save_full_graph_folder + f"uniformity_voc_{args.vocab_size}_ctxLen_{args.context_length}_stride{args.stride}_{args.perc_stories}%Wiki.jpg", precDone = 100, ctx_len = [t for t in range(1, args.context_length + 1)])
