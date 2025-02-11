"""
Download, preprocess and serve the WikiText dataset as a DataLoader.
"""

import argparse
import os
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from Trie_dataloader import create_dataloader, Trie_module_protV1
import time
import json
import random

os.environ["TOKENIZERS_PARALLELISM"] = "false"

print("Importing Done")


# -----------------------------------------------------------------------------
# CLI for constructing the dataset

SIZE_NODE_BYTES = 56 


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


if __name__ == "__main__":
    """
    These stages are designed to be run in order.

    To tokenize data with the Llama 2 tokenizer:
    """
    parser = argparse.ArgumentParser()
    parser.add_argument("--vocab_size", type=int, default=2048, help="pretokenization vocab size. 0 = use Llama 2 tokenizer.")
    parser.add_argument("--context_length", type=int, default=16, help="Context Length")
    parser.add_argument("--group", type=int, default=0, help="Used for sockeye purposes")
    parser.add_argument("--batch_size", type=int, default=5, help="Batch size")
    parser.add_argument("--stride", type=int, default=1, help="Window stride size")
    parser.add_argument("--perc_stories", type=int, default=100, help="percentage of stories")
    parser.add_argument("--num_tokens_to_proc", type=int, default=0, help="Number of tokens to process, if zero we go with the precentage")
    parser.add_argument("--scheduler_type", type=str, default="cosine", help="lr-scheduling style")
    parser.add_argument("--num_epochs", type=int, default=90, help="Step size")
    parser.add_argument("--LoadTrieFromFile", type=bool, default=False, help="Load from existing file")
    parser.add_argument("--num_bins", type=int, default=20, help="Number of shard")
    parser.add_argument("--root_ctx_len", type=int, default=2, help="Size of the root context lenght, shards will each have context length of (context_length - root_ctx_len) for the Trie")
    parser.add_argument("--Trie_dir", type=str, default="/scratch/st-cthrampo-1/vaalaa/NTP_LLM_DataLoader_Trie_TinyStories_V2048/Tries/voc2048_ctxLen12_100%TS_Stride1_NumBins20", help="Save Trie File name")
    args = parser.parse_args()

    
    # Set the current working directory to the expected path
    local_bin_folder_path = "/dev/shm/my_job/voc2048_ctxLen12_100%TS_Stride1_NumBins20"

    # Step 4: Load and Tokenize the Wikitext-2 Dataset
    # Example usage with stride
    print("_" * 100)
    print("Creating dataloader ... ")
    dataloader, vocab_size = create_dataloader(
        f'/scratch/st-cthrampo-1/vaalaa/TinyStories_tokenizer_bin/data/processed/train_{args.vocab_size}.bin',
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
    

    dataloader.dataset.initialize_shard_system(
        base_trie_path = local_bin_folder_path + "/",
        mapping_dir = local_bin_folder_path,
        transitions_dir = local_bin_folder_path + "/root",
        num_shards = args.num_bins
    )
    print("Complete!")
    print("_" * 100)

    thing = 0
    for X in dataloader:
        if thing == 1:
            break 
        thing += 1
    print("_" * 100)
    print(X)
    print("_" * 100)


    # Run several batches and time them
    num_batches = 10
    total_time = 0

    print(f"\nTesting {num_batches} batches with batch_size={args.batch_size}")
    print("-" * 50)

    for i in range(3):
        # start_idx = random.randint(0, 2000 - args.batch_size)
        # print(start_idx)
        
        start_time = time.time()
        batch = dataloader.dataset.get_batch_with_labels_DEBUG(i, args.batch_size)
        end_time = time.time()
        
        batch_time = end_time - start_time
        total_time += batch_time
        
        print(f"Batch {i+1}: {batch_time:.4f} seconds")
        print(f"  Sequences shape: {batch.sequences.shape}")
        print(f"  Hard targets shape: {batch.hard_targets.shape}")
        print(f"  Soft targets shape: {batch.soft_targets.shape}")

    avg_time = total_time / num_batches
    print("\nTiming Summary:")
    print(f"Average batch time: {avg_time:.4f} seconds")
    print(f"Throughput: {args.batch_size/avg_time:.2f} sequences/second")


    