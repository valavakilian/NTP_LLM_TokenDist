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
    parser.add_argument("--vocab_size", type=int, default=8196, help="pretokenization vocab size. 0 = use Llama 2 tokenizer.")
    parser.add_argument("--context_length", type=int, default=32, help="Context Length")
    parser.add_argument("--group", type=int, default=0, help="Used for sockeye purposes")
    parser.add_argument("--batch_size", type=int, default=512, help="Batch size")
    parser.add_argument("--stride", type=int, default=1, help="Window stride size")
    parser.add_argument("--perc_stories", type=int, default=100, help="percentage of stories")
    parser.add_argument("--num_tokens_to_proc", type=int, default=0, help="Number of tokens to process, if zero we go with the precentage")
    parser.add_argument("--scheduler_type", type=str, default="cosine", help="lr-scheduling style")
    parser.add_argument("--num_epochs", type=int, default=90, help="Step size")
    parser.add_argument("--LoadTrieFromFile", type=bool, default=False, help="Load from existing file")
    parser.add_argument("--num_bins", type=int, default=4, help="Number of shard")
    parser.add_argument("--root_ctx_len", type=int, default=2, help="Size of the root context lenght, shards will each have context length of (context_length - root_ctx_len) for the Trie")
    parser.add_argument("--Trie_dir", type=str, default="/scratch/st-cthrampo-1/vaalaa/NTP_LLM_DataStats_Trie_MultiProcessor_Wiki/Tries/", help="Save Trie File name")
    parser.add_argument("--tokenizer", type=str, choices=["bpe", "unigram", "wordpiece", "sentencepiece", "word", "charbpe", "hierarchical_bpe"], required=True, help="Type of tokenizer to use.")
    args = parser.parse_args()

    bin_folder_path = args.Trie_dir + f"shard{args.group}/"
    
    local_bin_folder_path = "./Trie_info/"
    if not os.path.exists(local_bin_folder_path):
        os.mkdir(local_bin_folder_path)
    print(f"Directory exists: {os.path.exists(local_bin_folder_path)}")
    bin_assigned_indices = np.load(bin_folder_path + 'indices.npy')
    valid_indices = np.loadtxt(bin_folder_path + 'shuffled_indices_locations.txt', dtype=int)
    print(valid_indices[0:100])


    # Step 4: Load and Tokenize the Wikitext-2 Dataset
    # Example usage with stride
    print("_" * 100)
    print("Creating dataloader ... ")
    dataloader, vocab_size = create_dataloader(
        f'/scratch/st-cthrampo-1/vaalaa/TinyStories_tokenizer_bin/data/processed/train_{args.tokenizer}_{args.vocab_size}.bin',
        context_length=args.context_length,
        batch_size=args.batch_size,
        data_percentage=args.perc_stories,
        token_pairs=bin_assigned_indices,
        valid_indices = valid_indices,
        stride=args.stride,   
        is_root = False, 
        root_ctx_len = 2
    )
    print("Complete!")
    print("_" * 100)
    args.vocab_size = vocab_size

    print("Running experiments for Vocab Size " + str(args.vocab_size) + " with Context Lenght " + str(args.context_length))
    

    print("_" * 100)
    print("Creating Trie ... ")
    # Get the dataset object
    dataset = dataloader.dataset

    # Create and get the Trie with custom batch size
    context_tree = dataset.create_and_get_trie(
        trie_path=local_bin_folder_path + f"Trie{args.group}_MT", 
        initial_size_gb=160,
        batch_size=args.batch_size  # Specify your preferred batch size
    )
    print("Complete!")
    print("_" * 100)

    print("_" * 100)
    print("Get shard stats")
    start_time = time.time()
    stats = context_tree.get_level_statistics()
    stat_time = time.time() - start_time
    print("Complete!")
    print(f"Took: {stat_time} seconds")
    print("_" * 100)

    with open(args.Trie_dir + f'stat_graphs/stats_shard{args.group}.json', 'w') as f:
        json.dump(stats, f)
    with open(args.Trie_dir + f'stat_graphs/stats_shard{args.group}.json', 'r') as f:
        loaded_dict = json.load(f)
    print("loaded_dict:")
    print(loaded_dict)

    