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
from collections import defaultdict

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



def read_binary_file(filepath):
    # Read tokens from binary file (uint16 format)
    return np.fromfile(filepath, dtype=np.uint16).tolist()

def get_context_distributions(sequence, context_len=5):
    # Initialize counts
    context_counts = defaultdict(lambda: defaultdict(int))
    context_totals = defaultdict(int)

    for i in range(0, len(sequence) - context_len):
        window = sequence[i:i + context_len + 1]

        for length in range(1, context_len + 1):
            context = tuple(window[0:length])
            target = window[length]

            context_counts[context][target] += 1
            context_totals[context] += 1
    
    # Convert counts to probabilities
    distributions = {}
    for context, next_token_counts in context_counts.items():
        total = context_totals[context]
        distributions[context] = {
            token: count/total 
            for token, count in next_token_counts.items()
        }
    
    return distributions


def get_soft_labels(sequence, distributions, vocab_size=5):
    seq_len = len(sequence)
    # Initialize output tensor: [seq_len, vocab_size]
    soft_labels = torch.zeros(seq_len, vocab_size)
    
    # For each position in sequence
    for pos in range(seq_len):
        # Try all possible context lengths ending at this position
        for context_len in range(1, pos + 2):
            # Get context
            start_idx = pos - context_len + 1
            if start_idx < 0:
                continue
                
            context = tuple(sequence[start_idx:pos + 1])
            
            # If we have this context, add its probabilities
            if context in distributions:
                for token, prob in distributions[context].items():
                    soft_labels[pos, token] = prob
                # Once we find a valid context, we use its probabilities
                break

    return soft_labels


def get_batch_soft_labels(batch_sequences, distributions, vocab_size=5):
    batch_size, seq_len = batch_sequences.shape
    # Initialize output tensor: [batch_size, seq_len, vocab_size]
    batch_soft_labels = torch.zeros(batch_size, seq_len, vocab_size)
    
    for b in range(batch_size):
        batch_soft_labels[b] = get_soft_labels(batch_sequences[b], distributions)
    
    return batch_soft_labels



def iterate_context_windows(dataset, batch_size=32):
    """
    Iterator that yields batches of context windows from the dataset.
    
    Args:
        dataset: TokenizedDataset instance
        batch_size: Number of context windows to yield at a time
        
    Yields:
        torch.Tensor: Batch of context windows of shape (batch_size, context_length)
    """
    num_windows = dataset.getNumWindows()
    
    for start_idx in range(0, num_windows, batch_size):
        # Get the end index for this batch
        end_idx = min(start_idx + batch_size, num_windows)
        current_batch_size = end_idx - start_idx
        
        # Create batch
        batch = torch.stack([dataset[i] for i in range(start_idx, end_idx)])
        
        # Slice off the last token (target) and keep only the context
        contexts = batch[:, :-1]
        
        yield contexts




def iterate_full_windows(dataset, batch_size=32):
    """
    Iterator that yields batches of context windows from the dataset.
    
    Args:
        dataset: TokenizedDataset instance
        batch_size: Number of context windows to yield at a time
        
    Yields:
        torch.Tensor: Batch of context windows of shape (batch_size, context_length)
    """
    num_windows = dataset.getNumWindows()
    
    for start_idx in range(0, num_windows, batch_size):
        # Get the end index for this batch
        end_idx = min(start_idx + batch_size, num_windows)
        current_batch_size = end_idx - start_idx
        
        # Create batch
        batch = torch.stack([dataset[i] for i in range(start_idx, end_idx)])
        
        # Slice off the last token (target) and keep only the context
        contexts = batch[:, :]
        
        yield contexts




def build_transition_distributions(dataset, batch_size=32):
    """
    Builds transition probability distributions for single tokens and token pairs.
    
    Args:
        dataset: TokenizedDataset instance
        batch_size: Size of batches to process at once
    
    Returns:
        dict: Contains two keys 'singles' and 'pairs', each mapping to a dict of 
             next-token probability distributions
    """
    # Initialize counters using nested defaultdict
    singles_counts = defaultdict(lambda: defaultdict(int))
    pairs_counts = defaultdict(lambda: defaultdict(int))
    
    # Process all contexts
    for contexts in iterate_context_windows(dataset, batch_size):
        for ctx_index in range(contexts.shape[0]):
            # Get single token and its next token
            single = int(contexts[ctx_index][0])
            single_next = int(contexts[ctx_index][1])
            singles_counts[single][single_next] += 1
            
            # Get token pair and its next token
            pair = tuple(contexts[ctx_index][0:2].tolist())
            pair_next = int(contexts[ctx_index][2])
            pairs_counts[pair][pair_next] += 1
    
    # Convert counts to probability distributions
    distributions = {}
    
    # Process singles
    for token, next_tokens in singles_counts.items():
        total = sum(next_tokens.values())
        distributions[token] = {
            next_token: count 
            for next_token, count in next_tokens.items()
        }
    
    # Process pairs
    for pair, next_tokens in pairs_counts.items():
        total = sum(next_tokens.values())
        distributions[pair] = {
            next_token: count 
            for next_token, count in next_tokens.items()
        }
    
    return distributions

def compare_nested_dicts(dict1, dict2):
    """
    Compare two nested dictionaries. If a sub-dictionary differs for a key, print both versions.

    Args:
        dict1 (dict): First dictionary of dictionaries.
        dict2 (dict): Second dictionary of dictionaries.
    """
    all_keys = dict1.keys() | dict2.keys()  # Union of keys from both dictionaries


    sum_dict_1 = 0
    sum_dict_2 = 0
    for key in all_keys:
        sub_dict1 = dict1.get(key, None)
        sub_dict2 = dict2.get(key, None)

        if type(key) == tuple:
            sum_dict_1 += sum(sub_dict1.values())
            sum_dict_2 += sum(sub_dict2.values())

        if sub_dict1 != sub_dict2:  # Only print if they differ
            print(f"Key '{key}' differs:")
            print(f"  from loader[{key}] = {sub_dict1}")
            print(f"  from sequence[{key}] = {sub_dict2}")
            print("-" * 50)
            input()
    
    print("Sum from loader: " + str(sum_dict_1))
    print("Sum from sequence: " + str(sum_dict_2))


def build_next_token_distributions(dataset, batch_size=32):
    """
    Build next token probability distributions for all context lengths 1 to context_length.
    Returns dict[context_tuple -> dict[next_token -> probability]]
    """
    counts = defaultdict(lambda: defaultdict(int))
    
    for windows in iterate_full_windows(dataset, batch_size):
        for i in range(windows.shape[0]):
            full_window = windows[i,:]
            
            # Build distributions for all possible context lengths
            for length in range(1, len(full_window)):
                # Get the context of current length
                curr_context = tuple(full_window[0:length].tolist())  # Take last 'length' tokens
                target = int(full_window[length])
                counts[curr_context][target] += 1
    
    # Convert counts to probabilities
    distributions = {}
    for context, next_tokens in counts.items():
        total = sum(next_tokens.values())
        distributions[context] = {
            token: count/total for token, count in next_tokens.items()
        }
    
    return distributions



if __name__ == "__main__":
    """
    These stages are designed to be run in order.

    To tokenize data with the Llama 2 tokenizer:
    """
    parser = argparse.ArgumentParser()
    parser.add_argument("--vocab_size", type=int, default=2048, help="pretokenization vocab size. 0 = use Llama 2 tokenizer.")
    parser.add_argument("--context_length", type=int, default=16, help="Context Length")
    parser.add_argument("--group", type=int, default=0, help="Used for sockeye purposes")
    parser.add_argument("--batch_size", type=int, default=3, help="Batch size")
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
    local_bin_folder_path = "/scratch/st-cthrampo-1/vaalaa/NTP_LLM_DataLoader_Trie_TinyStories_V2048/DEBUG/Trie_info"

    # Step 4: Load and Tokenize the Wikitext-2 Dataset
    # Example usage with stride
    print("_" * 100)
    print("Creating dataloader ... ")
    dataloader, vocab_size = create_dataloader(
        f'/scratch/st-cthrampo-1/vaalaa/TinyStories_tokenizer_bin/synthetic_data/synthetic_tokens.bin',
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


    print("Here we will analyze the dataset ... ")
    count_contexts = 0
    Root_softLabel_dict = dataloader.dataset.analyze_token_transitions("/scratch/st-cthrampo-1/vaalaa/NTP_LLM_DataLoader_Trie_TinyStories_V2048/DEBUG/Trie_info/root/", True)
    singles_pairs_pyDict = build_transition_distributions(dataloader.dataset)
    # compare_nested_dicts(Root_softLabel_dict, singles_pairs_pyDict)
    # input()

    # Process all contexts
    # for contexts in iterate_context_windows(dataloader.dataset, args.batch_size):
    #     print(contexts)
    #     # input()

    # print("Root_softLabel_dict: " + str(Root_softLabel_dict))
    # print("singles_pairs_pyDict: " + str(singles_pairs_pyDict))
    # print("count_contexts: " + str(count_contexts))
    # input()
    

    dataloader.dataset.initialize_shard_system(
        base_trie_path = local_bin_folder_path + "/",
        mapping_dir = local_bin_folder_path,
        transitions_dir = local_bin_folder_path + "/root",
        num_shards = args.num_bins
    )
    print("Complete!")
    print("_" * 100)
    # input()

    # thing = 0
    # for X in dataloader:
    #     if thing == 1:
    #         break 
    #     thing += 1
    # print("_" * 100)
    # print(X)
    # print("_" * 100)


    # Run several batches and time them
    num_batches = 2
    total_time = 0

    print(f"\nTesting {num_batches} batches with batch_size={args.batch_size}")
    print("-" * 50)


    filepath = "/scratch/st-cthrampo-1/vaalaa/TinyStories_tokenizer_bin/synthetic_data/synthetic_tokens.bin"
    sequence = read_binary_file(filepath)
    distributions = get_context_distributions(sequence)
    pairs_distr = {}
    for key in distributions.keys():
        if len(key) == 2:
            pairs_distr[key] = distributions[key]
        if len(key) == 1:
            pairs_distr[int(key[0])] = distributions[key]
    singles_pairs_pyDict_distr = {}
    for key in singles_pairs_pyDict.keys():
        singles_pairs_pyDict_distr[key] = {k:singles_pairs_pyDict[key][k] / sum(list(singles_pairs_pyDict[key].values())) for k in singles_pairs_pyDict[key].keys()}
    # compare_nested_dicts(pairs_distr, singles_pairs_pyDict_distr)
    print("^"*100)
    print("sequence: " + str(sequence))
    print("Len(sequence): " + str(len(sequence)))
    
    input()
    # print("distributions from sequence: ")
    # print(distributions)

    distributions_from_loader = build_next_token_distributions(dataloader.dataset, args.batch_size)
    # print("distributions from dataloader: ")
    # print(distributions_from_loader)
    compare_nested_dicts(distributions_from_loader, distributions)
    print("^"*100)
    input()

    any_dsitrepancy_detected = False
    for i in range(194):
        # start_idx = random.randint(0, 2000 - args.batch_size)
        # print(start_idx)
        
        start_time = time.time()
        batch = dataloader.dataset.get_batch_with_labels_DEBUG(i, args.batch_size)
        end_time = time.time()
        
        batch_time = end_time - start_time
        total_time += batch_time
        
        print(f"Batch {i+1}: {batch_time:.4f} seconds")
        # print(f"  Sequences shape: {batch.sequences.shape}")
        # print(f"  Hard targets shape: {batch.hard_targets.shape}")
        # print(f"  Soft targets shape: {batch.soft_targets.shape}")

        batch_soft_labels_python = []
        for seq in batch.sequences:
            seq_tensor_prob = torch.zeros((args.context_length, vocab_size))
            for i in range(1,6):
                for k in distributions[tuple(seq[0:i].tolist())].keys():
                    seq_tensor_prob[i-1,k] = distributions[tuple(seq[0:i].tolist())][k]
            batch_soft_labels_python.append(seq_tensor_prob)

        # input("#"*50)
        # input("Ready to see outputs ...., press enter")

        # print("batch.sequences: " + str(batch.sequences))
        # print("batch.hard_targets: " + str(batch.hard_targets))
        # print("batch.soft_targets: " + str(batch.soft_targets))
        # print("batch_soft_labels_python: " + str(batch_soft_labels_python))

        soft_targets = dataloader.dataset.get_batch_soft_tokens(batch.sequences)
        # print("soft_targets: " + str(soft_targets))

        for i in range(0, len(batch_soft_labels_python)):
            # print(batch_soft_labels_python[i])
            # print(soft_targets[i,:,:])
            if not torch.isclose(soft_targets[i,:,:], batch_soft_labels_python[i], rtol=1e-3, atol=1e-3).all():
                print("*"*100)
                print("The two models did not return the same output ... ")
                print("Sequence: " + str(batch.sequences[i,:]))
                print("_/\\" * 30)
                print("soft_targets_Trie:")
                print(soft_targets)
                print("_/\\" * 30)
                print("soft_targets_Trie:")
                print(soft_targets)
                print("*"*100)
                input()
                any_dsitrepancy_detected = True
            
        # input("One batch done ...., press enter")


    print("_" * 100)
    if any_dsitrepancy_detected:
        print("Discrepancy was detecter :(")
    else:
        print("No Discrepancy was detecter :)")
    input("_" * 100)

    avg_time = total_time / num_batches
    print("\nTiming Summary:")
    print(f"Average batch time: {avg_time:.4f} seconds")
    print(f"Throughput: {args.batch_size/avg_time:.2f} sequences/second")


    