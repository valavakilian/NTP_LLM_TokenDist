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
import matplotlib.pyplot as plt


os.environ["TOKENIZERS_PARALLELISM"] = "false"

print("Importing Done")


# -----------------------------------------------------------------------------
# CLI for constructing the dataset


def merge_branch_statistics(root_stats, shard_stats_list):
    """
    Merge root statistics with multiple branch shard statistics.
    
    Args:
        root_stats (dict): Statistics for the root levels (levels 1,2)
        shard_stats_list (list): List of dictionaries containing branch shard statistics
    
    Returns:
        dict: Merged statistics
    """
    # Initialize the merged stats with the root stats
    merged = {
        'levels': root_stats['levels'].copy(),
        'total_entropies': root_stats['total_entropies'].copy(),
        'avg_entropies': root_stats['avg_entropies'].copy(),
        'total_uniformity': root_stats['total_uniformity'].copy(),
        'avg_uniformity': root_stats['avg_uniformity'].copy(),
        'avg_support_sizes': root_stats['avg_support_sizes'].copy(),
        'total_visits': root_stats['total_visits'].copy(),
        'unique_nodes': root_stats['unique_nodes'].copy(),
        'total_num_oneHots': root_stats['total_num_oneHots'].copy()
    }
    
    # Get the maximum level across all shards
    max_level = max(max(shard['levels']) for shard in shard_stats_list)
    
    # For each level beyond root levels
    for level in range(3, max_level + 1):
        # Initialize accumulators for weighted averages
        total_entropy = 0
        total_uniformity = 0
        total_visits_level = 0
        total_unique_nodes = 0
        total_num_oneHots = 0
        weighted_support_sizes = 0
        
        # Process each shard
        for shard in shard_stats_list:
            if level in shard['levels']:
                level_idx = shard['levels'].index(level)
                visits = shard['total_visits'][level_idx]
                
                # Accumulate weighted entropy and uniformity
                total_entropy += shard['total_entropies'][level_idx]
                total_uniformity += shard['total_uniformity'][level_idx]
                
                # Accumulate other metrics
                total_visits_level += visits
                total_unique_nodes += shard['unique_nodes'][level_idx]
                total_num_oneHots += shard['total_num_oneHots'][level_idx]
                weighted_support_sizes += shard['avg_support_sizes'][level_idx] * visits
        
        # Append level
        merged['levels'].append(level)
        
        # Append total entropy and calculate average
        merged['total_entropies'].append(total_entropy)
        merged['avg_entropies'].append(total_entropy / total_visits_level if total_visits_level > 0 else 0)
        
        # Append total uniformity and calculate average
        merged['total_uniformity'].append(total_uniformity)
        merged['avg_uniformity'].append(total_uniformity / total_visits_level if total_visits_level > 0 else 0)
        
        # Calculate and append average support sizes
        avg_support_size = weighted_support_sizes / total_visits_level if total_visits_level > 0 else 0
        merged['avg_support_sizes'].append(avg_support_size)
        
        # Append other metrics
        merged['total_visits'].append(total_visits_level)
        merged['unique_nodes'].append(total_unique_nodes)
        merged['total_num_oneHots'].append(total_num_oneHots)
    
    return merged



def plot_branch_statistics(stats, save_path, file_prefix):
    """
    Create and save plots for various branch statistics.
    
    Args:
        stats (dict): Dictionary containing the merged statistics
        save_path (str): Directory path where plots should be saved
    """
    # Create save directory if it doesn't exist
    os.makedirs(save_path, exist_ok=True)
    
    # Common plot settings
    plt.style.use('seaborn')
    marker_style = 'o-'
    figsize = (10, 6)
    
    # Plot 1: Average Entropy
    plt.figure(figsize=figsize)
    plt.plot(stats['levels'], stats['avg_entropies'], marker_style)
    plt.yscale('log')
    plt.title(f'Average Entropy per Level ({file_prefix})')
    plt.xlabel('Level')
    plt.ylabel('Average Entropy')
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(os.path.join(save_path, f'avg_entropy{file_prefix}.png'), dpi=300, bbox_inches='tight')
    plt.close()
    
    # Plot 2: Average Uniformity
    plt.figure(figsize=figsize)
    plt.plot(stats['levels'], stats['avg_uniformity'], marker_style)
    plt.yscale('log')
    plt.title(f'Average Uniformity per Level ({file_prefix})')
    plt.xlabel('Level')
    plt.ylabel('Average Uniformity')
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(os.path.join(save_path, f'avg_uniformity{file_prefix}.png'), dpi=300, bbox_inches='tight')
    plt.close()
    
    # Plot 3: Average Support Sizes
    plt.figure(figsize=figsize)
    plt.plot(stats['levels'], stats['avg_support_sizes'], marker_style)
    plt.title(f'Average Support Sizes per Level ({file_prefix})')
    plt.xlabel('Level')
    plt.ylabel('Average Support Size')
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(os.path.join(save_path, f'avg_support_sizes{file_prefix}.png'), dpi=300, bbox_inches='tight')
    plt.close()
    
    # Plot 4: Total Visits
    plt.figure(figsize=figsize)
    plt.plot(stats['levels'], stats['total_visits'], marker_style)
    plt.title(f'Total Visits per Level ({file_prefix})')
    plt.xlabel('Level')
    plt.ylabel('Total Visits')
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(os.path.join(save_path, f'total_visits{file_prefix}.png'), dpi=300, bbox_inches='tight')
    plt.close()
    
    # Plot 5: OneHots Ratio
    onehot_ratio = [oh/tv for oh, tv in zip(stats['total_num_oneHots'], stats['total_visits'])]
    plt.figure(figsize=figsize)
    plt.plot(stats['levels'], onehot_ratio, marker_style)
    plt.title(f'Ratio of OneHots to Total Visits per Level ({file_prefix})')
    plt.xlabel('Level')
    plt.ylabel('OneHots / Total Visits')
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(os.path.join(save_path, f'onehot_ratio{file_prefix}.png'), dpi=300, bbox_inches='tight')
    plt.close()
    
    # Plot 6: Unique Nodes
    plt.figure(figsize=figsize)
    plt.plot(stats['levels'], stats['unique_nodes'], marker_style)
    plt.title(f'Unique Nodes per Level ({file_prefix})')
    plt.xlabel('Level')
    plt.ylabel('Number of Unique Nodes')
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(os.path.join(save_path, f'unique_nodes{file_prefix}.png'), dpi=300, bbox_inches='tight')
    plt.close()




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
    parser.add_argument("--num_bins", type=int, default=4, help="Step size")
    parser.add_argument("--root_ctx_len", type=int, default=2, help="Size of the root context lenght, shards will each have context length of (context_length - root_ctx_len) for the Trie")
    parser.add_argument("--Trie_dir", type=str, default="/scratch/st-cthrampo-1/vaalaa/NTP_LLM_DataStats_Trie_MultiProcessor_Wiki/Tries/", help="Save Trie File name")
    args = parser.parse_args()


    with open(args.Trie_dir + 'stat_graphs/stats_root.json', 'r') as f:
        root_stats = json.load(f)
    
    shard_stats_list = []
    for shard in range(0, args.num_bins):
        with open(args.Trie_dir + f'stat_graphs/stats_shard{shard}.json', 'r') as f:
            shard_stats_list.append(json.load(f))


    merged_stats = merge_branch_statistics(root_stats, shard_stats_list)
    with open(args.Trie_dir + 'stat_graphs/stats_merged.json', 'w') as f:
        json.dump(merged_stats, f)
    print(merged_stats)
    plot_branch_statistics(merged_stats, args.Trie_dir + "stat_graphs/", file_prefix=f"(voc = {args.vocab_size}, {args.num_tokens_to_proc} tokens)")


