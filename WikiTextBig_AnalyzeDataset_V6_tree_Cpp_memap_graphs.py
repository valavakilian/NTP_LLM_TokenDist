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
import numpy as np

from datasets import load_from_disk
from tokenizers import Tokenizer, models, trainers, pre_tokenizers
import torch
from torch.utils.data import Dataset, DataLoader
import os

import matplotlib.pyplot as plt

    
if __name__ == "__main__":
    """
    These stages are designed to be run in order.

    To tokenize data with the Llama 2 tokenizer:
    """

    vocab_sizes = [64, 128, 256, 512, 1024, 2048, 4096, 8196, 16392]
    # context_lenghts = [32, 64, 128, 256, 512, 1024]
    context_lenghts = [32]


    vocab_source = "custom"
    AR_training = True
    save_logs_folder =  "/scratch/st-cthrampo-1/vaalaa/NTP_LLM_TokenDist_WikiBig_childDict/logs_trees_cpp/"


    try:

        seq_eval_counter = 0


        data_log = {
            "entropy": {},
            "entropy_per_ctx_len": {},
            "num_total_ctx": {},
            "num_unique_ctx": {},
            "num_unique_ctx_len_list": {},
            "num_total_ctx_len_list": {},
            "insert_calc_time": {},
            "entropy_calc_time": {}
        }

        entropies_for_vocab = {}
        entropies_per_ctx_seen = {}

        for vocab_size in vocab_sizes:
            entropies_for_vocab[vocab_size] = {}
            # entropies_per_ctx_seen[vocab_size] = {}

            for ctx_len in context_lenghts:
                
                save_logs_filename = f"voc_{vocab_size}_ctxLen_{ctx_len}.pkl"

                with open(save_logs_folder + save_logs_filename, 'rb') as pickle_file:
                    data_log = pickle.load(pickle_file)

                entropies_for_vocab[vocab_size][ctx_len] = list(data_log["entropy"].values())[-1]
                
                print("_" * 100)
                print("vocab_size = " + str(vocab_size) + " ctx_len = " + str(ctx_len) + " :")
                print(data_log["entropy"])
                print(len(data_log["entropy"].values()))
                label = f"V = {vocab_size}, T = {ctx_len}"
                entropies_per_ctx_seen[label] = data_log["entropy"]
                # if len(data_log["entropy"].values()) == 99:
                #     entropies_per_ctx_seen[label] = data_log["entropy"]
                #     print("vocab_size = " + str(vocab_size) + " ctx_len = " + str(ctx_len) + " is fully done ")

        
        # print(entropies_for_vocab)
        # input()

        colors = ["#FF7043", "#FF5252", "#FF4081", "#E91E63", "#D81B60", "#C2185B", "#AD1457", "#880E4F", "#6A1B9A"]
        colors_dict = {vocab_sizes[i]: colors[i] for i in range(0, len(colors))}
        plt.figure(figsize=(8, 5))
        for vocab_size in vocab_sizes:
            plt.plot(context_lenghts, list(entropies_for_vocab[vocab_size].values()), color = colors_dict[vocab_size], label = "Voc = " + str(vocab_size))
        plt.xlabel('Context Length')
        plt.ylabel('Final Entropy')
        plt.title('Entropy vs vocab size and context length')
        plt.grid(True)
        plt.yscale("log")
        plt.legend()
        plt.savefig("/scratch/st-cthrampo-1/vaalaa/NTP_LLM_TokenDist_WikiBig_childDict/graphs_trees_cpp_postprocess/entropy_vs_voc_ctxLen.jpg")

                
        
        # print(len(entropies_per_ctx_seen.keys()))
        ctx_len_32 = {label: entropies_per_ctx_seen[label] for label in list(entropies_per_ctx_seen.keys()) if "T = 32" in label}
        colors = ["#FF7043", "#FF5252", "#FF4081", "#E91E63", "#D81B60", "#C2185B", "#AD1457", "#880E4F", "#6A1B9A"]
        colors_dict = {list(ctx_len_32.keys())[i]: colors[i] for i in range(0, len(list(ctx_len_32.keys())))}
        plt.figure(figsize=(8, 5))
        for scenario in ctx_len_32.keys():
            plt.plot(list(ctx_len_32[scenario].keys()), list(ctx_len_32[scenario].values()), color =colors_dict[scenario], label = scenario)
        plt.xlabel('Contexts seen')
        plt.ylabel('Final Entropy')
        plt.title('Entropy vs contexts seen for v = 64')
        plt.grid(True)
        plt.yscale("log")
        plt.xscale("log")
        plt.legend()
        plt.savefig("/scratch/st-cthrampo-1/vaalaa/NTP_LLM_TokenDist_WikiBig_childDict/graphs_trees_cpp_postprocess/entropy_ctxLen64_vs_vocSize_for_ctxSeen.jpg")

                
                


    except RuntimeError as e:
        print(f"An error occurred: {e}")


        

    